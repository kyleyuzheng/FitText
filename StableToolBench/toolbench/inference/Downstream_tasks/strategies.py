"""
Dynamic tool retrieval strategies: Single Pass, DBD, Scattershot, Memetic, and Memetic+ToolRet.

Abstractions:
rapidapi.py <--- strategies.py <--- services.py <--- prompts.py
"""
import logging
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Set

import numpy as np
from toolbench.inference.LLM.tokens import (
    FUNC_DESC_PATTERN,
    FUNC_DESC_PATTERN_TOLERANT,
)

_log = logging.getLogger(__name__)
from toolbench.retrieval.services import (
    retrieve_rapidapi_tools,
    normalize_api_descs,
    build_refine_or_regen_messages,
)
from toolbench.inference.LLM.prompts import (
    build_memetic_crossover_messages,
    build_memetic_mutation_messages,
    build_memetic_seed_messages,
    build_scattershot_seeded_messages,
    build_memetic_judge_messages,
    build_memetic_guided_mutation_messages,
)

# ---------- helpers ----------

def _only_blocks(text: str) -> List[str]:
    return [m.strip() for m in FUNC_DESC_PATTERN.findall(text or "") if m and m.strip()]


def _only_blocks_tolerant(text: str) -> List[str]:
    """Like :func:`_only_blocks` but accepts BEGIN without matching END.

    Used at strategy entry points where ``text`` is the raw agent output and
    newer tool-eager reasoning models may omit the closing marker. The
    strict variant remains the source of truth for internal memetic LLM
    outputs because those calls run with ``tools=[]`` and stay well-formed.
    """
    return [m.strip() for m in FUNC_DESC_PATTERN_TOLERANT.findall(text or "") if m and m.strip()]

def _extract_single_block(text: str):
    matches = FUNC_DESC_PATTERN.findall(text or "")
    if not matches:
        return None
    for m in matches:
        s = (m or "").strip()
        if s:
            return s
    return None

def _api_list_to_keys(api_list, lineage_index: int | None = None) -> List[Dict[str, Any]]:
    """Convert query_json['api_list'] into plain API key dicts."""
    out = []
    for it in (api_list or []):
        d = {
            "category_name": it["category_name"],
            "tool_name":     it["tool_name"],
            "api_name":      it["api_name"],
        }
        if lineage_index is not None:
            d["lineage_index"] = lineage_index
        out.append(d)
    return out

def _jaccard_similarity(s1: str, s2: str) -> float:
    """Token-based Jaccard similarity for memory penalty."""
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    denominator = len(set1 | set2)
    return intersection / denominator


def _threadsafe_parse(llm, messages, tools=None, process_id=0, **kwargs):
    """Thread-safe LLM call that bypasses conversation_history state."""
    from toolbench.inference.LLM.chatgpt_function_model import chat_completion_request
    response = chat_completion_request(
        key=llm.openai_key,
        base_url=llm.base_url,
        messages=messages,
        tools=tools or None,
        model=llm.model,
        process_id=process_id,
        **kwargs,
    )
    try:
        usage = response.get("usage") or {}
        total_tokens = usage.get("total_tokens", 0)
        return response["choices"][0]["message"], 0, total_tokens
    except Exception:
        return {"role": "assistant", "content": ""}, -1, 0

# ---------- genetic / memetic ----------

def run_memetic_strategy(llm_output: str, wrapper, llm):
    """
    Memetic retrieval strategy (population=5, generations=3 default).

    Sequential by design — each lineage runs its population evaluation /
    crossover / mutation / refinement loop in a single thread. The
    intended optimization is per-ancestor parallelism layered on top of this
    function.

    Structure (v1):
    1. Population Init: Per-ancestor seeding via retrieval-grounded LLM generation.
    2. Loop:
       a. Selection & Crossover (Global Search)
       b. Mutation
       c. Local Search (LLM Refinement - if memetic) -> Matches DBD logic
       d. Fitness Evaluation (Multi-objective: Quality - MemoryPenalty)
       e. Survival
    """
    
    #
    # The flow ensures we refine offspring BEFORE evaluation to capture the "Memetic" gain.

    # 1. Parameters & Setup
    population_size = max(2, int(getattr(wrapper, "population_size", 6)))
    max_generations = max(1, int(getattr(wrapper, "generation_num", 3)))
    similarity_threshold = float(getattr(wrapper, "similarity_threshold", 0.95))
    is_memetic = getattr(wrapper, "memetic", False)
    top_k_child = int(getattr(wrapper, "retrieved_api_nums", 10))
    final_tool_budget = int(getattr(wrapper, "final_tool_budget", top_k_child))

    # ── Cross-model memetic (2026-05-25) ───────────────────────────────────
    # ``llm`` is the DFSDT backbone (passed via DFS.py → wrapper.dynamic_retrieve_base_on_des).
    # When ``wrapper.refine_llm`` is set (--refine_model differs from --chatgpt_model),
    # every evolution-subprocess LLM call below (seed-population, crossover, mutation,
    # refinement) routes through ``refine_llm`` instead. Sentinel: None → fall back to ``llm``.
    # This keeps the ancestor pseudo-tool generation (DFSDT) on one model while the
    # memetic evolution loop runs on a different model — useful when the backbone
    # is a reasoning-style model (Responses API) and the evolution loop is a faster
    # non-reasoning model (Chat Completions), or vice versa. The CLI / model pins
    # are the source of truth for which models are in play; nothing here.
    refine_llm = getattr(wrapper, "refine_llm", None) or llm

    # Access tool_memory safely (it might be on the wrapper if passed from DFS, or empty)
    tool_memory = getattr(wrapper, "tool_memory", {})
    # Ensure we are looking at keys if it is a dict
    memory_intents = tool_memory.keys() if isinstance(tool_memory, dict) else tool_memory

    # Tolerant extraction at the agent entry — tool-eager reasoning models can omit
    # the closing marker (see DFS.py 2026-05-25 patch). Internal memetic
    # extractions below still use _only_blocks (strict) because those LLM
    # calls run with tools=[] and the outputs are well-formed.
    ancestors = _only_blocks_tolerant(llm_output)
    if not ancestors:
        ancestors = [wrapper.input_description or ""]

    # ------------------------------------------------------------------
    # 2. Fitness — pluggable via wrapper.fitness_method or $FITNESS_FUNCTION.
    #
    # The default "baseline_jaccard" path mirrors the legacy inline closure
    # bit-for-bit (Jaccard over wrapper.tool_memory keys + retrieval term).
    # Alternative methods (smc, agm, dpp) re-score on a per-query
    # ``BeliefHistory`` constructed live as the population is evaluated.
    # See the offline fitness-comparison utilities for the design rationale.
    # ------------------------------------------------------------------
    _fitness_method = (
        getattr(wrapper, "fitness_method", None)
        or os.environ.get("FITNESS_FUNCTION", "baseline_jaccard")
    )
    # Per-query belief history shared across lineages.
    _belief_history = None
    _scorer = None
    if _fitness_method != "baseline_jaccard":
        # Lazy import; only needed when an alt fitness is requested. Keeps the
        # baseline path zero-cost.
        try:
            from toolbench.inference.fitness import (  # type: ignore
                BeliefCandidate,
                BeliefHistory,
                BaselineJaccardFitness,
                SMCFitness,
                AGMFitness,
                DPPFitness,
            )
            from toolbench.inference.instrumentation.embedder import (  # type: ignore
                SharedEmbedder,
            )
        except Exception as _imp_err:
            _log.warning(
                "memetic: fitness=%s requested but imports failed (%s); "
                "falling back to baseline_jaccard.",
                _fitness_method,
                _imp_err,
            )
            _fitness_method = "baseline_jaccard"
        else:
            _scorer = {
                "smc": SMCFitness,
                "agm": AGMFitness,
                "dpp": DPPFitness,
            }[_fitness_method]()
            _belief_history = BeliefHistory()
            _embedder = SharedEmbedder.instance()

    def _retrieval_distribution(scores: List[float], top_k_target: int = 32) -> "np.ndarray":
        """Softmax over the top-K retrieval scores (proxy for full-catalog dist).

        We softmax only the top-K=top_k_target sims (which is what ``mt`` already
        gives us) — this is an approximation of the full-catalog retrieval
        distribution used by the pluggable fitness scorers.  Empirically, the full
        catalog softmax mass concentrates on the top-K anyway because most
        per-query catalog tools have near-zero sim to the belief; a top-K
        softmax preserves the high-density region that drives SMC / AGM
        likelihoods and KL terms.
        """
        if not scores:
            return np.zeros(1, dtype=np.float32)
        arr = np.asarray(scores[:top_k_target], dtype=np.float32)
        arr = arr - float(np.max(arr))
        e = np.exp(arr)
        s = float(np.sum(e))
        if s <= 0.0:
            return np.full(arr.shape, 1.0 / arr.size, dtype=np.float32)
        return (e / s).astype(np.float32)

    def calculate_fitness(pseudotool: str, retrieval_results: List[dict]) -> float:
        """Score one belief against the running query population.

        Routes to a pluggable scorer based on ``_fitness_method``. The default
        ``baseline_jaccard`` path is the legacy v1 closure verbatim — Jaccard
        penalty against ``wrapper.tool_memory`` keys plus 0.7*top1 + 0.3*top3
        retrieval term.  Alt methods (smc/agm/dpp) build a per-query
        BeliefHistory and emit scorer-specific scalars on the (belief, prior)
        pair; the history is mutated after each call so subsequent candidates
        see the prior population.

        Args:
            pseudotool: The belief text being scored.
            retrieval_results: Top-K retrieval results from
                ``retrieve_rapidapi_tools`` (list of dicts with ``score``).

        Returns:
            Scalar fitness score (higher = better).
        """
        if not retrieval_results:
            return 0.0
        scores = [(s.get("score") or 0.0) for s in retrieval_results]

        # ----- baseline_jaccard path (legacy production closure) -------
        if _fitness_method == "baseline_jaccard" or _scorer is None:
            retrieval_score = (0.7 * float(scores[0])) + (
                0.3 * (sum(scores[:3]) / min(3, len(scores)))
            )
            penalty = 0.0
            if memory_intents:
                similarities = [
                    _jaccard_similarity(pseudotool, mem) for mem in memory_intents
                ]
                max_sim = max(similarities) if similarities else 0.0
                # Soft gate: only penalize if similarity is statistically
                # significant (mirrors strategies.py:152 legacy behaviour).
                if max_sim > 0.3:
                    penalty = 0.5 * max_sim
            return retrieval_score - penalty

        # ----- alt fitness path (smc / agm / dpp) ----------------------
        try:
            emb = _embedder.encode([pseudotool])  # (1, D), L2-normalised
            belief_vec = (
                emb[0].astype(np.float32)
                if emb.size > 0
                else np.zeros(1, dtype=np.float32)
            )
        except Exception as _enc_err:
            _log.warning(
                "memetic: embedder failed (%s) — falling back to baseline_jaccard for this call.",
                _enc_err,
            )
            retrieval_score = (0.7 * float(scores[0])) + (
                0.3 * (sum(scores[:3]) / min(3, len(scores)))
            )
            return retrieval_score

        dist = _retrieval_distribution(scores)
        top_k_ids = [
            f"{(r.get('category') or '')}/"
            f"{(r.get('tool_name') or '')}/"
            f"{(r.get('api_name') or '')}"
            for r in retrieval_results
        ]
        cand = BeliefCandidate(  # type: ignore[name-defined]
            qid=str(getattr(wrapper, "query_id", "")),
            belief_idx=len(_belief_history.prior_beliefs),
            belief_text=pseudotool,
            belief_embedding=belief_vec,
            retrieval_distribution=dist,
            retrieval_top_k_ids=top_k_ids[:10],
            retrieval_top_k_scores=[float(s) for s in scores[:10]],
        )
        try:
            score_val = float(_scorer.score(candidate=cand, history=_belief_history))
        except Exception as _score_err:
            _log.warning(
                "memetic: scorer %s failed (%s) — defaulting to retrieval_score.",
                _fitness_method,
                _score_err,
            )
            score_val = (0.7 * float(scores[0])) + (
                0.3 * (sum(scores[:3]) / min(3, len(scores)))
            )
        # Update history AFTER scoring so the next candidate sees this one.
        _belief_history.add(cand)
        return score_val

    # Evolution-subprocess temperature — resolved once per call to
    # ``run_memetic_strategy`` and shared by all evolutionary LLM call sites
    # (seed-population generation, mutation, crossover, LLM refinement).
    #
    # Resolution chain (first non-None wins):
    #   1. ``wrapper.evolution_temperature``  — explicit override from the
    #      RunConfig (``cfg.model.evolution_temperature``), the new public
    #      surface for the trigger-vs-evolution split (2026-05-24).
    #   2. ``wrapper.base_temp``              — legacy attribute used by all
    #      231 existing memetic result dirs; preserved for back-compat so
    #      old provenance still reproduces bit-identical.
    #   3. ``1.5`` (default)                  — production default. Empirical
    #      basis: existing ``memetic_p5_g3_temp1.5`` partial-n cells on
    #      the legacy comparison solver hit 17–29% on the hard-question subset where the
    #      T=0.9 path was 0%.
    _evo_temp = (
        getattr(wrapper, "evolution_temperature", None)
        if getattr(wrapper, "evolution_temperature", None) is not None
        else (
            getattr(wrapper, "base_temp", None)
            if getattr(wrapper, "base_temp", None) is not None
            else 1.5
        )
    )

    # 3. Helpers (Aligned with other strategies)
    def llm_refine(pseudotool: str, retrieved_tools: List[dict], ancestor_anchor: str) -> str:
        """Uses the standard DBD refinement prompt logic.

        Runs at the evolutionary-subprocess temperature ``_evo_temp`` (memetic
        local-search step). See the resolution chain above for the lookup.
        """
        exemplars = normalize_api_descs(retrieved_tools)
        # We treat this as a "refinement" turn (turn 1 logic from DBD)
        msgs = build_refine_or_regen_messages(
            mode="dbd",
            query=wrapper.input_description,
            refinement=True,
            ancestor_desc=ancestor_anchor, # Context anchor per lineage
            current_desc=pseudotool,
            lineage_examples=exemplars,
            turn=1
        )
        msg, _, _ = refine_llm.parse_with_messages(
            messages=msgs,
            tools=[],
            process_id=getattr(wrapper, "process_id", 0),
            temperature=_evo_temp,
        )
        refined = _extract_single_block((msg or {}).get("content", ""))
        return refined or pseudotool

    def generate_population_from_ancestor(ancestor_desc: str, exemplar_blurbs: List[str], size: int) -> List[str]:
        """Seed-population generation — runs at ``_evo_temp`` (evolution subprocess).

        Despite the name, this call is INSIDE the memetic evolution loop and
        therefore routes through ``refine_llm``: the ancestor pseudo-tool
        comes from DFSDT; everything from population generation onward
        belongs to the evolution model.
        """
        exemplar_block = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(exemplar_blurbs) if str(s).strip()
        )
        msgs = build_memetic_seed_messages(
            query=wrapper.input_description,
            ancestor_desc=ancestor_desc,
            exemplar_block=exemplar_block,
            population_size=size,
        )
        msg, _, _ = refine_llm.parse_with_messages(
            messages=msgs,
            tools=[],
            process_id=getattr(wrapper, "process_id", 0),
            temperature=_evo_temp,
        )
        pop = _only_blocks((msg or {}).get("content", ""))

        if not pop:
            pop = []

        # Ensure we have exactly 'size' individuals; pad with ancestor if needed.
        while len(pop) < size:
            pop.append(ancestor_desc)

        return pop[:size]

    retrieval_iterations: List[dict] = []
    api_keys: List[Dict[str, Any]] = []
    final_descriptions: List[str] = []
    lineage_summaries: List[Dict[str, Any]] = []

    # Run memetic search per ancestor lineage
    for lineage_index, ancestor in enumerate(ancestors):
        # Seed retrieval to collect exemplars for this ancestor
        qj_seed, mt_seed = retrieve_rapidapi_tools(
            retriever=wrapper.retriever,
            query=ancestor,
            top_k=wrapper.retrieved_api_nums,
            tool_root_dir=wrapper.tool_root_dir,
        )
        retrieval_iterations.append({
            "phase": "seed_retrieval",
            "lineage_index": lineage_index,
            "ancestor_description": ancestor,
            "retrieved_tools": mt_seed,
        })

        exemplar_blurbs = normalize_api_descs(mt_seed)
        exemplar_block = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(exemplar_blurbs) if str(s).strip()
        )
        population = generate_population_from_ancestor(ancestor, exemplar_blurbs, population_size)
        best_solution = {
            "pseudotool": population[0] if population else ancestor,
            "score": float("-inf"),
            "qj": {"api_list": []},
            "mt": []
        }
        last_scored_population: List[Tuple[str, float, dict, List[dict]]] = []

        generations_ran = 0

        # 4. Evolution Loop (per lineage)
        for generation in range(1, max_generations + 1):
            generations_ran = generation
            # --- Evaluation Phase ---
            scored_population = []
            for individual in population:
                qj, mt = retrieve_rapidapi_tools(
                    retriever=wrapper.retriever,
                    query=individual,
                    top_k=top_k_child,
                    tool_root_dir=wrapper.tool_root_dir,
                )
                score = calculate_fitness(individual, mt)
                scored_population.append((individual, score, qj, mt))
                
                # Track best
                if score > best_solution["score"]:
                    best_solution = {"pseudotool": individual, "score": score, "qj": qj, "mt": mt}

                retrieval_iterations.append({
                    "phase": "evaluation",
                    "generation": generation,
                    "lineage_index": lineage_index,
                    "candidate_pseudotool": individual,
                    "score": score,
                    "top_tool_score": (mt[0].get("score") if mt else 0)
                })

            # FIX 1: Assignment moved outside the per-individual loop
            last_scored_population = scored_population

            # Check termination
            if best_solution["score"] >= similarity_threshold:
                break
            if generation == max_generations:
                break

            # --- Selection (Elitism + Top K) ---
            # Sort by fitness desc
            scored_population.sort(key=lambda x: x[1], reverse=True)
            # Elitism: Keep top 1 unchanged
            next_gen = [scored_population[0][0]]

            parents_pool = [x[0] for x in scored_population[:population_size // 2]]

            # --- Crossover & Mutation (LLM-driven) ---
            while len(next_gen) < population_size:
                use_crossover = len(parents_pool) >= 2 and random.random() < 0.5

                if use_crossover:
                    p1, p2 = random.sample(parents_pool, 2)
                    messages = build_memetic_crossover_messages(
                        query=wrapper.input_description,
                        ancestor_desc=ancestor,
                        parent1_desc=p1,
                        parent2_desc=p2,
                        exemplar_block=exemplar_block,
                    )
                    fallback = p1
                else:
                    parent = random.choice(parents_pool)
                    messages = build_memetic_mutation_messages(
                        query=wrapper.input_description,
                        ancestor_desc=ancestor,
                        parent_desc=parent,
                        exemplar_block=exemplar_block,
                    )
                    fallback = parent

                # Crossover / mutation — evolutionary subprocess call site.
                msg, _, _ = refine_llm.parse_with_messages(
                    messages=messages,
                    tools=[],
                    process_id=getattr(wrapper, "process_id", 0),
                    temperature=_evo_temp,
                )
                child = _extract_single_block((msg or {}).get("content", "")) or fallback

                next_gen.append(child)

            # --- Memetic Phase: Local Search on Offspring ---
            # If Memetic, we refine the children *before* they enter the next evaluation cycle
            if is_memetic:
                refined_gen = []
                top_k_refine = int(getattr(wrapper, "top_k_refine", getattr(wrapper, "retrieved_api_nums", 10)))
                for child in next_gen:
                    # Skip refinement for elites
                    if child == scored_population[0][0]:
                        refined_gen.append(child)
                        continue

                    _, mt_child = retrieve_rapidapi_tools(
                        retriever=wrapper.retriever,
                        query=child,
                        top_k=top_k_refine,
                        tool_root_dir=wrapper.tool_root_dir,
                    )
                    refined_child = llm_refine(child, mt_child, ancestor)
                    refined_gen.append(refined_child)

                    retrieval_iterations.append({
                        "phase": "memetic_refine",
                        "generation": generation,
                        "lineage_index": lineage_index,
                        "original": child,
                        "refined": refined_child,
                        "retrieved_tools": mt_child,
                    })
                population = refined_gen
            else:
                population = next_gen

        # 5. Finalize lineage
        best = best_solution["pseudotool"]

        if hasattr(wrapper, "tool_memory") and isinstance(wrapper.tool_memory, set):
            wrapper.tool_memory.add(ancestor)
            wrapper.tool_memory.add(best)

        elif hasattr(wrapper, "tool_memory") and isinstance(wrapper.tool_memory, dict):
            wrapper.tool_memory[ancestor] = True
            wrapper.tool_memory[best] = True

        # --- Population-level voting for tools ---
        # FIX 2: Ensures we read last_scored_population from the final generation evaluated
        votes: Dict[Tuple[str, str, str], int] = defaultdict(int)
        ranks: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        scores: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        allowed: Set[Tuple[str, str, str]] = set()

        for it in (qj_seed.get("api_list") or []):
            allowed.add((it["category_name"], it["tool_name"], it["api_name"]))

        for individual, _, qj_eval, mt_eval in last_scored_population:
            for it in (qj_eval.get("api_list") or []):
                allowed.add((it["category_name"], it["tool_name"], it["api_name"]))

            for rank_idx, item in enumerate((mt_eval or [])[:top_k_child]):
                key = (item["category"], item["tool_name"], item["api_name"])
                votes[key] += 1
                ranks[key].append(rank_idx)
                scores[key].append(item.get("score"))

        def _avg(xs, default):
            xs2 = [v for v in xs if v is not None]
            return (sum(xs2) / len(xs2)) if xs2 else default

        def _sort_key(k):
            return (-votes[k], _avg(ranks[k], 9999.0), -_avg(scores[k], -1e9))

        winning_keys = sorted(votes.keys(), key=_sort_key)
        winning_keys = [k for k in winning_keys if k in allowed]
        
        # FIX 3: Bound winning keys by final_tool_budget
        winning_keys = winning_keys[:final_tool_budget]

        # Fallback: ensure we return something if voting failed to produce enough keys
        if not winning_keys:
            for it in (qj_seed.get("api_list") or []):
                key = (it["category_name"], it["tool_name"], it["api_name"])
                if key not in winning_keys:
                    winning_keys.append(key)
                if len(winning_keys) == final_tool_budget:
                    break

        for c, t, a in winning_keys:
            api_keys.append({"category_name": c, "tool_name": t, "api_name": a, "lineage_index": lineage_index})

        retrieval_iterations.append({
            "phase": "population_vote",
            "lineage_index": lineage_index,
            "winning_keys": [
                {"category_name": c, "tool_name": t, "api_name": a}
                for c, t, a in winning_keys
            ],
            "votes": {str(k): votes[k] for k in votes},
        })

        final_descriptions.append(best_solution["pseudotool"])
        lineage_summaries.append({
            "ancestor": ancestor,
            "seed_exemplars": exemplar_blurbs,
            "best_description": best_solution["pseudotool"],
            "best_score": best_solution["score"],
            "generations_ran": generations_ran,
            "winning_keys": [
                {"category_name": c, "tool_name": t, "api_name": a}
                for c, t, a in winning_keys
            ],
        })

    payload = {
        "strategy": "memetic" if is_memetic else "genetic",
        "final_descriptions": final_descriptions,
        "lineages": lineage_summaries,
        "population_size": population_size,
        "generation_num": max_generations,
    }

    return api_keys, retrieval_iterations, payload

# ---------- Scattershot ----------

def run_scattershot_strategy(llm_output: str, wrapper, llm):
    """
    Per-lineage Scattershot:
      • Seed retrieval per ancestor (local exemplars) -> qj_seed (for allowed keys) & mt_seed (for logs)
      • Generate 'size' children; retrieve each with top_k = retrieved_api_nums
      • Vote using child mt lists (rank/score), but only allow winners present in the combined seed/child qj api_lists
      • Tie-break: votes desc, avg rank asc, avg score desc
      • Truncate to retrieved_api_nums winners per lineage
      • Return resolvable API keys + trace
    """
    from itertools import chain

    retrieval_iterations: List[dict] = []
    size = max(1, int(getattr(wrapper, "scattershot_size", 5)))
    top_k_child = int(getattr(wrapper, "retrieved_api_nums", 15))

    ancestors = _only_blocks(llm_output)

    api_keys: List[Dict[str, Any]] = []

    # seed per ancestor
    seeds_qj: List[Dict[str, Any]] = []
    seeds_mt: List[List[Dict[str, Any]]] = []
    exemplars_per_ancestor: List[List[str]] = []

    for ai, ancestor in enumerate(ancestors):
        qj_seed, mt_seed = retrieve_rapidapi_tools(
            retriever=wrapper.retriever,
            query=ancestor,
            top_k=top_k_child,
            tool_root_dir=wrapper.tool_root_dir,
        )
        retrieval_iterations.append({
            "phase": "seed_retrieval",
            "ancestor_index": ai,
            "ancestor_description": ancestor,
            "retrieved_tools": mt_seed,
        })
        seeds_qj.append(qj_seed)
        seeds_mt.append(mt_seed)
        exemplars_per_ancestor.append(normalize_api_descs(mt_seed))

    # children per lineage
    lineage_summaries: List[Dict[str, Any]] = []

    for ai, (ancestor, exemplar_blurbs, qj_seed, mt_seed) in enumerate(
        zip(ancestors, exemplars_per_ancestor, seeds_qj, seeds_mt)
    ):
        votes: Dict[tuple, int] = defaultdict(int)
        ranks: Dict[tuple, List[int]] = defaultdict(list)
        scores: Dict[tuple, List[float]] = defaultdict(list)

        # Resolvable winners must come from the combined seed/child qj api_lists.
        allowed: set[tuple] = set()
        for it in (qj_seed.get("api_list") or []):
            allowed.add((it["category_name"], it["tool_name"], it["api_name"]))

        qjs_children: List[Dict[str, Any]] = []

        exemplar_block = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(exemplar_blurbs) if str(s).strip()
        )

        child_descriptions: List[str] = []

        # --- Parallel scattershot: generate + retrieve all children concurrently ---
        def _scatter_child(si):
            msgs = build_scattershot_seeded_messages(
                query=wrapper.input_description,
                ancestor_desc=ancestor,
                exemplar_block=exemplar_block,
            )
            msg, _, _ = _threadsafe_parse(
                llm, msgs, process_id=wrapper.process_id, temperature=getattr(wrapper, "base_temp", 0.9)
            )
            child_desc = _extract_single_block((msg or {}).get("content", ""))
            if not child_desc:
                return si, None, None, None
            qj_child, mt_child = retrieve_rapidapi_tools(
                retriever=wrapper.retriever,
                query=child_desc,
                top_k=top_k_child,
                tool_root_dir=wrapper.tool_root_dir,
            )
            return si, child_desc, qj_child, mt_child

        with ThreadPoolExecutor(max_workers=size) as pool:
            scatter_results = list(pool.map(_scatter_child, range(size)))

        for si, child_desc, qj_child, mt_child in scatter_results:
            if child_desc is None:
                retrieval_iterations.append({
                    "phase": "child_gen_empty",
                    "ancestor_index": ai,
                    "sample_index": si,
                })
                continue
            retrieval_iterations.append({
                "phase": "child_retrieval",
                "ancestor_index": ai,
                "sample_index": si,
                "child_description": child_desc,
                "retrieved_tools": mt_child,
            })
            qjs_children.append(qj_child)
            child_descriptions.append(child_desc)
            for it in (qj_child.get("api_list") or []):
                allowed.add((it["category_name"], it["tool_name"], it["api_name"]))

            for rank_idx, item in enumerate((mt_child or [])[:top_k_child]):
                key = (item["category"], item["tool_name"], item["api_name"])
                votes[key] += 1
                ranks[key].append(rank_idx)
                scores[key].append(item.get("score"))

        def _avg(xs, default):
            xs2 = [v for v in xs if v is not None]
            return (sum(xs2) / len(xs2)) if xs2 else default

        def _sort_key(k):
            return (-votes[k], _avg(ranks[k], 9999.0), -_avg(scores[k], -1e9))

        lineage_keys = sorted(votes.keys(), key=_sort_key)
        # keep only resolvable winners
        lineage_keys = [k for k in lineage_keys if k in allowed]
        # truncate
        top_keys = lineage_keys[:top_k_child]

        # backfill in stable order if quota not met
        if len(top_keys) < top_k_child:
            for it in chain(qj_seed.get("api_list") or [], *[q.get("api_list") or [] for q in qjs_children]):
                key = (it["category_name"], it["tool_name"], it["api_name"])
                if key not in top_keys:
                    top_keys.append(key)
                    if len(top_keys) == top_k_child:
                        break

        for c, t, a in top_keys:
            api_keys.append({"category_name": c, "tool_name": t, "api_name": a, "lineage_index": ai})

        lineage_summaries.append({
            "ancestor": ancestor,
            "seed_exemplars": exemplar_blurbs,
            "generated_children": child_descriptions,
            "winning_keys": [
                {"category_name": c, "tool_name": t, "api_name": a}
                for c, t, a in top_keys
            ],
        })

    payload = {
        "strategy": "scattershot",
        "lineages": lineage_summaries,
        "ancestors": ancestors,
        "scattershot_size": size,
    }

    return api_keys, retrieval_iterations, payload

# ---------- single-pass ----------

def run_single_pass_strategy(llm_output: str, wrapper) -> Tuple[List[Dict[str, Any]], List[dict], Dict[str, Any]]:
    # Tolerant: agent may omit the closing marker.
    descriptions = _only_blocks_tolerant(llm_output)
    retrieval_iterations: List[dict] = []
    api_keys: List[Dict[str, Any]] = []

    for i, desc in enumerate(descriptions):
        qj, mt = retrieve_rapidapi_tools(
            retriever=wrapper.retriever,
            query=desc,
            top_k=wrapper.retrieved_api_nums,
            tool_root_dir=wrapper.tool_root_dir,
        )
        retrieval_iterations.append({
            "iteration": 1,
            "lineage_index": i,
            "current_description": desc,
            "retrieved_tools": mt,
        })
        api_keys.extend(_api_list_to_keys(qj["api_list"], lineage_index=i))

    payload = {
        "strategy": "single_pass",
        "final_descriptions": descriptions,
    }

    return api_keys, retrieval_iterations, payload

# ---------- DBD ----------

def run_dbd_strategy(llm_output: str, wrapper, llm) -> Tuple[List[Dict[str, Any]], List[dict], Dict[str, Any]]:
    # Tolerant: agent may omit the closing marker.
    ancestors = _only_blocks_tolerant(llm_output)
    retrieval_iterations: List[dict] = []
    if not ancestors:
        payload = {
            "strategy": "dbd",
            "lineages": [],
            "final_descriptions": [],
        }
        return [], retrieval_iterations, payload

    turns = max(0, getattr(wrapper, "dbd_refine_turns", 0))
    final_descs: List[str] = []

    lineage_summaries: List[Dict[str, Any]] = []

    for lineage_index, ancestor in enumerate(ancestors):
        current = ancestor
        examples: List[str] = []
        seen = set()

        if turns == 0:
            qj, mt = retrieve_rapidapi_tools(
                retriever=wrapper.retriever,
                query=current,
                top_k=wrapper.retrieved_api_nums,
                tool_root_dir=wrapper.tool_root_dir,
            )
            retrieval_iterations.append({
                "iteration": 1,
                "lineage_index": lineage_index,
                "ancestor_description": ancestor,
                "current_description": current,
                "retrieved_tools": mt,
            })
            final_descs.append(current)
            lineage_summaries.append({
                "ancestor": ancestor,
                "final_description": current,
                "examples_used": examples[:],
            })
            continue

        for t in range(1, turns + 1):
            qj, mt = retrieve_rapidapi_tools(
                retriever=wrapper.retriever,
                query=current,
                top_k=wrapper.retrieved_api_nums,
                tool_root_dir=wrapper.tool_root_dir,
            )
            retrieval_iterations.append({
                "iteration": t,
                "lineage_index": lineage_index,
                "ancestor_description": ancestor,
                "current_description": current,
                "retrieved_tools": mt,
            })
            for blurb in normalize_api_descs(mt):
                if blurb not in seen:
                    seen.add(blurb); examples.append(blurb)

            msgs = build_refine_or_regen_messages(
                mode="dbd",
                query=wrapper.input_description,
                refinement=getattr(wrapper, "refinement", False),
                ancestor_desc=ancestor,
                current_desc=current,
                lineage_examples=examples,
                turn=t,
                refine_style=getattr(wrapper, "refine_style", "fittext"),
            )
            
            msg, _, _ = llm.parse_with_messages(messages=msgs, tools=[], process_id=getattr(wrapper, "process_id", 0))
            content = (msg or {}).get("content", "") or ""
            next_desc = _extract_single_block(content)
            if next_desc:
                current = next_desc

        final_descs.append(current)
        lineage_summaries.append({
            "ancestor": ancestor,
            "final_description": current,
            "examples_used": examples[:],
        })

    # final retrieval on stabilized lineages
    api_keys: List[Dict[str, Any]] = [] 
    for i, desc in enumerate(final_descs):
        qj, mt = retrieve_rapidapi_tools(
            retriever=wrapper.retriever,
            query=desc,
            top_k=wrapper.retrieved_api_nums,
            tool_root_dir=wrapper.tool_root_dir,
        )
        retrieval_iterations.append({
            "iteration": "final",
            "lineage_index": i,
            "current_description": desc,
            "retrieved_tools": mt,
        })
        api_keys.extend(_api_list_to_keys(qj["api_list"], lineage_index=i))

    payload = {
        "strategy": "dbd",
        "lineages": lineage_summaries,
        "final_descriptions": final_descs,
        "turns_per_lineage": turns,
    }

    return api_keys, retrieval_iterations, payload


# ---------- memetic toolret ----------

def run_memetic_toolret_strategy(llm_output: str, wrapper, llm):
    """
    Memetic ToolRet — a self-contained variant of memetic v1 designed for
    standalone tool retrieval without any DFSDT scaffolding.

    Differences from run_memetic_strategy:
    - Ancestors come from wrapper.input_description, not llm_output blocks.
    - No tool_memory: fitness is pure retrieval score (no memory penalty).
    - Refinement always runs (no is_memetic flag).
    - Never reads from or writes to wrapper.tool_memory.
    """
    population_size   = max(2, int(getattr(wrapper, "population_size", 6)))
    max_generations   = max(1, int(getattr(wrapper, "generation_num", 3)))
    similarity_threshold = float(getattr(wrapper, "similarity_threshold", 0.95))
    top_k_child       = int(getattr(wrapper, "retrieved_api_nums", 10))
    final_tool_budget = int(getattr(wrapper, "final_tool_budget", top_k_child))

    # Single ancestor: the original query — no DFSDT block parsing.
    ancestors = [wrapper.input_description or ""]

    # Cross-model memetic — see run_memetic_strategy for full rationale.
    # ``llm`` is the caller-provided LLM; ``refine_llm`` overrides it for
    # every evolution-subprocess call below when wrapper.refine_llm is set.
    refine_llm = getattr(wrapper, "refine_llm", None) or llm

    # Evolution-subprocess temperature lookup chain (mirrors run_memetic_strategy):
    #   evolution_temperature → base_temp → 1.5
    _evo_temp = (
        getattr(wrapper, "evolution_temperature", None)
        if getattr(wrapper, "evolution_temperature", None) is not None
        else (
            getattr(wrapper, "base_temp", None)
            if getattr(wrapper, "base_temp", None) is not None
            else 1.5
        )
    )

    # ── Fitness: retrieval score only (no memory penalty) ──────────────────
    def calculate_fitness(pseudotool: str, retrieval_results: List[dict]) -> float:
        if not retrieval_results:
            return 0.0
        scores = [s.get("score") or 0.0 for s in retrieval_results]
        return (0.7 * float(scores[0])) + (0.3 * (sum(scores[:3]) / min(3, len(scores))))

    # ── LLM helpers ────────────────────────────────────────────────────────
    def llm_refine(pseudotool: str, retrieved_tools: List[dict], anchor: str) -> str:
        """LLM refinement — evolutionary subprocess call site (T=_evo_temp)."""
        exemplars = normalize_api_descs(retrieved_tools)
        msgs = build_refine_or_regen_messages(
            mode="dbd",
            query=wrapper.input_description,
            refinement=True,
            ancestor_desc=anchor,
            current_desc=pseudotool,
            lineage_examples=exemplars,
            turn=1,
        )
        msg, _, _ = refine_llm.parse_with_messages(
            messages=msgs,
            tools=[],
            process_id=getattr(wrapper, "process_id", 0),
            temperature=_evo_temp,
        )
        return _extract_single_block((msg or {}).get("content", "")) or pseudotool

    def generate_population(ancestor: str, exemplar_blurbs: List[str], size: int) -> List[str]:
        """Seed-population generation — evolutionary subprocess call site (T=_evo_temp)."""
        exemplar_block = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(exemplar_blurbs) if str(s).strip()
        )
        msgs = build_memetic_seed_messages(
            query=wrapper.input_description,
            ancestor_desc=ancestor,
            exemplar_block=exemplar_block,
            population_size=size,
        )
        msg, _, _ = refine_llm.parse_with_messages(
            messages=msgs,
            tools=[],
            process_id=getattr(wrapper, "process_id", 0),
            temperature=_evo_temp,
        )
        pop = _only_blocks((msg or {}).get("content", ""))
        while len(pop) < size:
            pop.append(ancestor)
        return pop[:size]

    retrieval_iterations: List[dict] = []
    api_keys: List[Dict[str, Any]] = []
    final_descriptions: List[str] = []
    lineage_summaries: List[Dict[str, Any]] = []

    for lineage_index, ancestor in enumerate(ancestors):
        # Seed retrieval for exemplars
        qj_seed, mt_seed = retrieve_rapidapi_tools(
            retriever=wrapper.retriever,
            query=ancestor,
            top_k=wrapper.retrieved_api_nums,
            tool_root_dir=wrapper.tool_root_dir,
        )
        retrieval_iterations.append({
            "phase": "seed_retrieval",
            "lineage_index": lineage_index,
            "ancestor_description": ancestor,
            "retrieved_tools": mt_seed,
        })

        exemplar_blurbs = normalize_api_descs(mt_seed)
        exemplar_block = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(exemplar_blurbs) if str(s).strip()
        )
        population = generate_population(ancestor, exemplar_blurbs, population_size)
        best_solution: Dict[str, Any] = {
            "pseudotool": population[0] if population else ancestor,
            "score": float("-inf"),
            "qj": {"api_list": []},
            "mt": [],
        }
        last_scored_population: List[Tuple[str, float, dict, List[dict]]] = []
        generations_ran = 0

        for generation in range(1, max_generations + 1):
            generations_ran = generation

            # Evaluation
            scored_population = []
            for individual in population:
                qj, mt = retrieve_rapidapi_tools(
                    retriever=wrapper.retriever,
                    query=individual,
                    top_k=top_k_child,
                    tool_root_dir=wrapper.tool_root_dir,
                )
                score = calculate_fitness(individual, mt)
                scored_population.append((individual, score, qj, mt))
                if score > best_solution["score"]:
                    best_solution = {"pseudotool": individual, "score": score, "qj": qj, "mt": mt}
                retrieval_iterations.append({
                    "phase": "evaluation",
                    "generation": generation,
                    "lineage_index": lineage_index,
                    "candidate_pseudotool": individual,
                    "score": score,
                    "top_tool_score": (mt[0].get("score") if mt else 0),
                })

            last_scored_population = scored_population

            if best_solution["score"] >= similarity_threshold or generation == max_generations:
                break

            # Selection (elitism + top-k)
            scored_population.sort(key=lambda x: x[1], reverse=True)
            next_gen = [scored_population[0][0]]
            parents_pool = [x[0] for x in scored_population[:population_size // 2]]

            # Crossover / mutation
            while len(next_gen) < population_size:
                use_crossover = len(parents_pool) >= 2 and random.random() < 0.5
                if use_crossover:
                    p1, p2 = random.sample(parents_pool, 2)
                    messages = build_memetic_crossover_messages(
                        query=wrapper.input_description,
                        ancestor_desc=ancestor,
                        parent1_desc=p1,
                        parent2_desc=p2,
                        exemplar_block=exemplar_block,
                    )
                    fallback = p1
                else:
                    parent = random.choice(parents_pool)
                    messages = build_memetic_mutation_messages(
                        query=wrapper.input_description,
                        ancestor_desc=ancestor,
                        parent_desc=parent,
                        exemplar_block=exemplar_block,
                    )
                    fallback = parent
                # Crossover / mutation — evolutionary subprocess call site (T=_evo_temp).
                msg, _, _ = refine_llm.parse_with_messages(
                    messages=messages,
                    tools=[],
                    process_id=getattr(wrapper, "process_id", 0),
                    temperature=_evo_temp,
                )
                next_gen.append(_extract_single_block((msg or {}).get("content", "")) or fallback)

            # Memetic local search — always on
            top_k_refine = int(getattr(wrapper, "top_k_refine", top_k_child))
            refined_gen = []
            for child in next_gen:
                if child == scored_population[0][0]:
                    refined_gen.append(child)
                    continue
                _, mt_child = retrieve_rapidapi_tools(
                    retriever=wrapper.retriever,
                    query=child,
                    top_k=top_k_refine,
                    tool_root_dir=wrapper.tool_root_dir,
                )
                refined_child = llm_refine(child, mt_child, ancestor)
                refined_gen.append(refined_child)
                retrieval_iterations.append({
                    "phase": "memetic_refine",
                    "generation": generation,
                    "lineage_index": lineage_index,
                    "original": child,
                    "refined": refined_child,
                    "retrieved_tools": mt_child,
                })
            population = refined_gen

        # Population-level voting
        votes: Dict[Tuple[str, str, str], int]         = defaultdict(int)
        ranks: Dict[Tuple[str, str, str], List[int]]   = defaultdict(list)
        scores: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
        allowed: Set[Tuple[str, str, str]] = set()

        for it in (qj_seed.get("api_list") or []):
            allowed.add((it["category_name"], it["tool_name"], it["api_name"]))

        for individual, _, qj_eval, mt_eval in last_scored_population:
            for it in (qj_eval.get("api_list") or []):
                allowed.add((it["category_name"], it["tool_name"], it["api_name"]))
            for rank_idx, item in enumerate((mt_eval or [])[:top_k_child]):
                key = (item["category"], item["tool_name"], item["api_name"])
                votes[key] += 1
                ranks[key].append(rank_idx)
                scores[key].append(item.get("score"))

        def _avg(xs, default):
            xs2 = [v for v in xs if v is not None]
            return (sum(xs2) / len(xs2)) if xs2 else default

        winning_keys = sorted(votes.keys(), key=lambda k: (-votes[k], _avg(ranks[k], 9999.0), -_avg(scores[k], -1e9)))
        winning_keys = [k for k in winning_keys if k in allowed][:final_tool_budget]

        if not winning_keys:
            for it in (qj_seed.get("api_list") or []):
                key = (it["category_name"], it["tool_name"], it["api_name"])
                if key not in winning_keys:
                    winning_keys.append(key)
                if len(winning_keys) == final_tool_budget:
                    break

        for c, t, a in winning_keys:
            api_keys.append({"category_name": c, "tool_name": t, "api_name": a, "lineage_index": lineage_index})

        retrieval_iterations.append({
            "phase": "population_vote",
            "lineage_index": lineage_index,
            "winning_keys": [{"category_name": c, "tool_name": t, "api_name": a} for c, t, a in winning_keys],
            "votes": {str(k): votes[k] for k in votes},
        })

        final_descriptions.append(best_solution["pseudotool"])
        lineage_summaries.append({
            "ancestor": ancestor,
            "seed_exemplars": exemplar_blurbs,
            "best_description": best_solution["pseudotool"],
            "best_score": best_solution["score"],
            "generations_ran": generations_ran,
            "winning_keys": [{"category_name": c, "tool_name": t, "api_name": a} for c, t, a in winning_keys],
        })

    payload = {
        "strategy": "memetic_toolret",
        "final_descriptions": final_descriptions,
        "lineages": lineage_summaries,
        "population_size": population_size,
        "generation_num": max_generations,
    }
    return api_keys, retrieval_iterations, payload


# ---------- just_query (zero-retrieval floor baseline) ----------

def run_just_query_strategy(llm_output: str, wrapper, llm) -> Tuple[List[Dict[str, Any]], List[dict], Dict[str, Any]]:
    """Zero-retrieval floor baseline — answer the query directly, no tools.

    This is the "everything off"
    comparator: the agent gets the query text only and must answer from prior
    knowledge alone. There is NO retrieval, NO tool catalog, NO second pass,
    NO critic. The point of the technique is to measure what the LLM can
    solve without any tool-retrieval assistance.

    Args:
        llm_output: Ignored (no upstream planning step is meaningful here;
            kept for router signature parity with the other strategies).
        wrapper: The strategy wrapper. Only ``input_description`` and
            ``process_id`` are read.
        llm: The LLM client. ``parse_with_messages`` is called once with
            ``tools=[]`` so the model receives no function schemas.

    Returns:
        Tuple of:
          - retrieved_tools: empty list — this technique retrieves nothing.
          - retrieval_iterations: a single-entry log of the LLM call (for
            manifest provenance / cost tracking).
          - metadata: ``{"strategy": "just_query", "answer": <text>}``.
            StableToolBench's pass-rate judge can score the ``answer`` field
            directly when no tool calls are made.
    """
    query = getattr(wrapper, "input_description", "") or ""

    # Single LLM call, no tools, no retrieval, no schema.
    messages = [
        {"role": "system",
         "content": "Answer the user's question directly to the best of your "
                    "knowledge. Do not call any tools."},
        {"role": "user",
         "content": f"Answer the following question: {query}"},
    ]
    msg, _, _ = llm.parse_with_messages(
        messages=messages,
        tools=[],
        process_id=getattr(wrapper, "process_id", 0),
    )
    answer = (msg or {}).get("content", "") or ""

    retrieval_iterations: List[dict] = [{
        "phase": "just_query",
        "query": query,
        "answer": answer,
    }]
    payload: Dict[str, Any] = {
        "strategy": "just_query",
        "answer": answer,
    }
    # Empty tool_ids — this is the point: zero-retrieval floor.
    return [], retrieval_iterations, payload


# ---------- router ----------

def select_and_run_strategy(llm_output: str, wrapper, llm) -> Tuple[List[Dict[str, Any]], List[dict], Dict[str, Any]]:
    # just_query is the zero-retrieval floor — must be checked FIRST so it
    # short-circuits every other branch (including the planner LLM step in
    # callers that gate on it). See run_just_query_strategy() docstring.
    if getattr(wrapper, "just_query", False):
        return run_just_query_strategy(llm_output, wrapper, llm)
    if getattr(wrapper, "dbd", False):
        return run_dbd_strategy(llm_output, wrapper, llm)
    if getattr(wrapper, "scattershot", False):
        return run_scattershot_strategy(llm_output, wrapper, llm)
    if getattr(wrapper, "memetic_toolret", False):
        return run_memetic_toolret_strategy(llm_output, wrapper, llm)
    if getattr(wrapper, "memetic", False):
        return run_memetic_strategy(llm_output, wrapper, llm)
    return run_single_pass_strategy(llm_output, wrapper)
