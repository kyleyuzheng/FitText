import argparse
import os
import sys
from pathlib import Path

# Ensure Toolret/ is importable both as a package (Toolret.*) AND as scripts run
# from this directory (bare 'retriever' / 'strategy' imports).
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

import json

# Load model pins — single source of truth for all model identifiers.
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "StableToolBench"))
from toolbench.observability.pins import load_pins as _load_pins  # noqa: E402
_pins = _load_pins(_repo_root)
_DEFAULT_PLAN_MODEL = _pins["agents"]["gpt_4_1"]
_DEFAULT_REFINE_MODEL = _pins["agents"]["gpt_4_1_mini"]

# Dataset config: code/web/customized/toolbench (production splits).
# ``mini`` is a synthetic test-fixture split — registered here so the runner
# adapter does not warn "Unknown ToolRet split 'mini'" when the E2E test
# fixture points at it.  In production the ``mini`` mapping is never used —
# the adapter takes the local-queries fast path when the queries_root
# override is set in cfg.extra.
dataset_categories = {
    "web": ["autotools-food", "autotools-music", "restgpt-tmdb", "autotools-weather", "restgpt-spotify", "toolbench", "toollens", "apibank",
            "mnms",  "reversechain", "tooleyes", "ultratool", "t-eval-dialog", "t-eval-step", "apigen", "rotbench", "taskbench-daily", "toolace", "toolemu"],
    "code": ["gorilla-pytorch", "gorilla-tensor", "gorilla-huggingface", "craft-tabmwp", "craft-vqa", "craft-math-algebra", "toolink"],
    "customized": ["gpt4tools", "taskbench-huggingface", "taskbench-multimedia", "toolbench-sam", "toolalpaca", "gta", "tool-be-honest", "appbench", "metatool"],
    "toolbench": ["toolbench"],
    "mini": ["mini"],
}

def cal_eval(qrels, results, k_values=(5, 10, 20)):
    """
    qrels:   Dict[str, Dict[str, int]]      # qid -> {doc_id: relevance}
    results: Dict[str, Dict[str, float]]    # qid -> {doc_id: score}
    """
    map_string = "map_cut." + ",".join(map(str, k_values))
    ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
    recall_string = "recall." + ",".join(map(str, k_values))
    precision_string = "P." + ",".join(map(str, k_values))

    import pytrec_eval

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)  # {qid: {"ndcg_cut_5": ..., "map_cut_5": ..., ...}}

    ndcg, _map, rec, prec, comp = {}, {}, {}, {}, {}
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        rec[f"Recall@{k}"] = 0.0
        prec[f"Precision@{k}"] = 0.0
        comp[f"Comprehensiveness@{k}"] = 0.0

    for qid, sc in scores.items():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += sc[f"ndcg_cut_{k}"]
            _map[f"MAP@{k}"]  += sc[f"map_cut_{k}"]
            rec[f"Recall@{k}"] += sc[f"recall_{k}"]
            prec[f"Precision@{k}"] += sc[f"P_{k}"]
            comp[f"Comprehensiveness@{k}"] += 1.0 if sc[f"recall_{k}"] == 1.0 else 0.0

    n_q = max(1, len(scores))
    def _norm(d): return {k: round(v / n_q, 5) for k, v in d.items()}

    metrics = {}
    for block in (_norm(ndcg), _norm(_map), _norm(rec), _norm(prec), _norm(comp)):
        metrics.update(block)

    return metrics, len(scores)

def add_run_results(results_for_qid, retrieved_tool_ids, retrieved_tool_scores, score_mode):
    """Populate a pytrec run while preserving strategy order when requested."""
    n_tools = len(retrieved_tool_ids)
    for rank, (tid, raw_score) in enumerate(zip(retrieved_tool_ids, retrieved_tool_scores)):
        tid = str(tid)
        if tid in results_for_qid:
            continue
        if score_mode == "rank":
            results_for_qid[tid] = float(n_tools - rank)
        else:
            results_for_qid[tid] = float(raw_score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="code", choices=["code", "web", "customized"],
                        help="Dataset config name (code/web/customized)")
    parser.add_argument("--corpus_path", type=str, default="./data/retrieval/Toolret",
                        help="Path to corpus JSONL (id, description).")
    parser.add_argument("--embedding_model_path", type=str, default="all-MiniLM-L6-v2",
                        help="Path to the embedding model.")
    parser.add_argument('--plan_llm_model', type=str, default=_DEFAULT_PLAN_MODEL,
                        help='Planner LLM. Default from configs/model_pins.yaml agents.gpt_4_1.')
    parser.add_argument("--refine_llm_model", type=str, default=_DEFAULT_REFINE_MODEL,
                        help='Refiner LLM. Default from configs/model_pins.yaml agents.gpt_4_1_mini.')
    parser.add_argument("--openai_key", type=str, default="", help="OpenAI API key.")
    parser.add_argument("--base_url_plan", type=str, default=None)
    parser.add_argument("--base_url_refine", type=str, default=None)
    parser.add_argument("--strategy", type=str, default="dbd", choices=["dbd", "scattershot", "single_pass", "just_query"])
    parser.add_argument("--retrieved_api_nums", type=int, default=5)
    parser.add_argument("--example_num", type=int, default=15)
    parser.add_argument("--dbd_refine_turns", type=int, default=3, help="Number of DBD refinement turns (if using DBD strategy)")
    parser.add_argument("--refinement", action='store_true', help="Whether to use refinement (DBD only), otherwise using regeneration")
    parser.add_argument("--scattershot_size", type=int, default=5,
                        help="Number of scattershot pseudo-tool samples. Default 5 is the paper recipe.")
    parser.add_argument("--detailed_result_path", type=str, default="./result")
    parser.add_argument("--result_path", type=str, default="./result/")
    parser.add_argument(
        "--score_mode",
        choices=["rank", "cosine_legacy"],
        default="rank",
        help=(
            "ToolRet emits an ordered retrieval list. Use strictly decreasing "
            "rank scores by default so pytrec_eval honors that order. "
            "This keeps the same NDCG metric and only changes run-file score "
            "projection. cosine_legacy reproduces older runs that let pytrec "
            "rerank by raw cosine."
        ),
    )

    args = parser.parse_args()

    try:
        from Toolret.retriever import ToolRetriever
        from Toolret.strategy.strategies import select_and_run_strategy, strategy_wrapper
    except ImportError:
        from retriever import ToolRetriever  # type: ignore[no-redef]
        from strategy.strategies import select_and_run_strategy, strategy_wrapper  # type: ignore[no-redef]
    from datasets import load_dataset
    from tqdm import tqdm

    detailed_result_path = os.path.join(args.detailed_result_path, args.dataset, args.strategy)
    os.makedirs(detailed_result_path, exist_ok=True)
    result_path = os.path.join(args.result_path, args.dataset, args.strategy, "overall_results.txt")
    corpus_path = os.path.join(args.corpus_path, args.dataset, "des_corpus.json")

    tool_retriever = ToolRetriever(corpus_path=corpus_path, model_path=args.embedding_model_path)
    wrapper = strategy_wrapper(
        strategy=args.strategy,
        retriever=tool_retriever,
        example_num=args.example_num,
        retrieved_api_nums=args.retrieved_api_nums,
        dbd_refine_turns=args.dbd_refine_turns,
        scattershot_size=args.scattershot_size,
        refinement=args.refinement,
        api_key=args.openai_key
    )
    qrels_all, results_all = {}, {}

    per_subset_metrics = {}
    per_subset_qcounts = {}

    with open(result_path, "w") as f:
        f.write("=== Per-subset metrics ===\n")
        f.write(f"score_mode={args.score_mode}\n")
        
    for subset_name in dataset_categories[args.dataset]:
        print(f"Processing subset: {subset_name}")
        subset_detailed_result_path = os.path.join(detailed_result_path, f"{subset_name}.jsonl")
        if os.path.exists(subset_detailed_result_path):
            print(f"Detailed result file {subset_detailed_result_path} exists, skipping subset {subset_name}.")
            continue
        
        sub_dataset = load_dataset("mangopy/ToolRet-Queries", subset_name)['queries']
        qrels = {}
        results = {}
        for sample in tqdm(sub_dataset, desc="Proccessing samples"):
            qid = str(sample['id'])
            query = sample['query']
            gt_tools = json.loads(sample['labels'])
            qrels[qid] = {str(x['id']): int(x['relevance']) for x in gt_tools}

            retrieved_tool_ids, retrieved_tool_descriptions, retrieved_tool_scores = select_and_run_strategy(query, wrapper, args.plan_llm_model, args.refine_llm_model, subset_detailed_result_path, args.base_url_plan, args.base_url_refine)

            ## Do the evaluation of the retrieved tools

            results[qid] = {}
            add_run_results(results[qid], retrieved_tool_ids, retrieved_tool_scores, args.score_mode)

        subset_metrics, qcount = cal_eval(qrels, results)
        per_subset_metrics[subset_name] = subset_metrics
        per_subset_qcounts[subset_name] = qcount

        line = f"{subset_name} {subset_metrics} (#queries={qcount})\n"
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        
        for qid, rels in qrels.items():
            qrels_all[qid] = rels
        for qid, res in results.items():
            results_all[qid] = res
    global_metrics, global_qcount = cal_eval(qrels_all, results_all)

    weighted_metrics = {}
    if per_subset_metrics:
        metric_names = list(next(iter(per_subset_metrics.values())).keys())
        total_q = sum(per_subset_qcounts.values()) or 1
        for m in metric_names:
            num = sum(per_subset_metrics[s][m] * per_subset_qcounts[s] for s in per_subset_metrics)
            weighted_metrics[m] = round(num / total_q, 5)
    # Save overall results
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "a") as f:
        f.write("\n=== Global metrics over all evaluated queries ===\n")
        f.write(f"{global_metrics} (#queries={global_qcount})\n")

        f.write("\n=== Weighted-avg over subsets (by #queries, optional) ===\n")
        f.write(f"{weighted_metrics}\n")

if __name__ == "__main__":
    main()
