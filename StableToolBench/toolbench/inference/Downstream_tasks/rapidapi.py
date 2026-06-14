import re
import os
import json
import time
from typing import Any, Dict, List

import requests
from tqdm import tqdm

from toolbench.inference.LLM.prompts import *
from toolbench.inference.LLM.tokens import *
from toolbench.inference.LLM.functions import *
from toolbench.inference.LLM.usage_tracker import reset_usage, snapshot_usage

from termcolor import colored
import random
from toolbench.inference.LLM.chatgpt_function_model import ChatGPTFunction
from toolbench.inference.LLM.tool_llama_model import ToolLLaMA
from toolbench.inference.LLM.retriever import ToolRetriever, RemoteToolRetriever

from toolbench.retrieval.resolver import (
    get_white_list, fetch_api_json, contain, build_tool_description, api_json_to_openai_json
)
from toolbench.retrieval.services import (
    retrieve_rapidapi_tools
)
from toolbench.inference.Downstream_tasks.strategies import select_and_run_strategy

from toolbench.inference.Algorithms.single_chain import single_chain
from toolbench.inference.Algorithms.DFS import DFS_tree_search
from toolbench.inference.server import get_rapidapi_response
from toolbench.utils import (
    standardize,
    change_name,
    replace_llama_with_condense
)

from toolbench.inference.Downstream_tasks.base_env import base_env


def build_chatgpt_function_llm(model, openai_key, base_url=None):
    """Build a :class:`ChatGPTFunction` for the given model, or ``None``.

    Single-responsibility helper for the cross-model memetic surface
    (``--chatgpt_model`` for DFSDT backbone vs ``--refine_model`` for the
    memetic evolution loop). The provider routing (Chat Completions vs
    Responses API vs vLLM) is handled inside ``ChatGPTFunction`` via the
    underlying ``make_client`` factory — this helper is just a thin
    None-guarded constructor so callers can write ``build_chatgpt_function_llm(...)
    or fallback_llm`` and get sentinel-style behaviour.

    Args:
        model: Full dated model tag, as pinned in ``configs/model_pins.yaml``
            (passed in via ``--chatgpt_model`` / ``--refine_model``). Falsy
            → returns ``None``. No defaults are baked in here — the CLI /
            pin file is the source of truth.
        openai_key: OpenAI API key. Forwarded verbatim.
        base_url: Optional API base URL override (vLLM-served models).

    Returns:
        A new ``ChatGPTFunction`` instance, or ``None`` when ``model`` is
        falsy. Identity-distinct from any other call site — callers MUST
        compare to ``None`` (sentinel), never compare by identity to a
        previously-built instance.
    """
    if not model:
        return None
    return ChatGPTFunction(model=model, openai_key=openai_key, base_url=base_url)


# rapidapi env wrapper
class rapidapi_wrapper(base_env):
    def __init__(self, query_json, tool_descriptions, retriever, args, process_id=0, query_id=None):
        super(rapidapi_wrapper,self).__init__()
        # originally was super(rapidapi_wrapper).__init__()
        self.lineage_memory = getattr(args, 'summarize_lineage_memory', False) # for DFSDT memory and lineage memory reconciliation
        self.memory_llm = getattr(args, 'lineage_summarizer', "o1-mini")

        self.dbd = getattr(args, 'dbd', False)
        self.refinement = getattr(args, 'refinement', False)   # <-- default False = regeneration

        self.dbd_refine_turns = getattr(args, 'dbd_refine_turns', 0)

        self.disable_midexec_retrieval = bool(getattr(args, 'disable_midexec_retrieval', False))
        self.refine_style = getattr(args, 'refine_style', 'fittext')
        # Root-node ablation bookkeeping (see DFS.py): False on the root io_state;
        # set True on a lineage's io_state once its (root) retrieval has run, so
        # mid-execution triggers can be gated off when disable_midexec_retrieval
        # is set. Inherited by descendant nodes via deepcopy.
        self._retrieval_done = False

        self.retrieve_mode = getattr(args, 'retrieve_mode', 'normal')
        self.retrieved_api_nums = getattr(args, 'retrieved_api_nums', 5)

        self.scattershot = getattr(args, 'scattershot', False)
        self.scattershot_size = getattr(args, 'size', 5)
        # Zero-retrieval floor baseline. select_and_run_strategy() checks this
        # FIRST (strategies.py) and routes to run_just_query_strategy, which
        # short-circuits retrieval + planner. Enabled via qa_pipeline --just_query.
        self.just_query = getattr(args, 'just_query', False)
        """
        Runtime guard, but this file should not really ever be run directly.
        Enforcement should be done in qa_pipeline_open_domain.py
        """
        if self.dbd and self.retrieve_mode != "dynamic":
            raise ValueError("DBD requires retrieve_mode='dynamic'.")
        if self.scattershot and self.dbd:
            raise ValueError("Scattershot is incompatible with DBD.")

        self.genetic = getattr(args, 'genetic', False)
        self.memetic = getattr(args, 'memetic', False)
        self.population_size = getattr(args, 'population_size', 6)
        self.generation_num = getattr(args, 'generation_num', 3)
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.95)
        self.memetic_toolret = getattr(args, 'memetic_toolret', False)
        self.base_temp = getattr(args, 'base_temp', 0.9)
        # Memetic evolution-subprocess temperature. None → fall back to base_temp.
        # Recommended split-temperature setting: base_temp=1.0 (trigger / DFSDT outer
        # loop), evolution_temperature=1.5 (memetic seed/mutation/crossover/refine rescue T).
        self.evolution_temperature = getattr(args, 'evolution_temperature', None)

        # ── Cross-model memetic (2026-05-25) ────────────────────────────────
        # The DFSDT backbone (``args.chatgpt_model``) generates the ancestor
        # pseudo-tool that seeds memetic; ``args.refine_model`` (if distinct)
        # drives every evolution-subprocess LLM call inside the memetic loop:
        # seed-population generation, crossover, mutation, and per-child
        # refinement. When ``refine_model`` is None OR equals
        # ``chatgpt_model``, the sentinel stays ``None`` and the strategy
        # falls back to the backbone llm — bit-identical single-model
        # behaviour.
        #
        # IMPORTANT: ``--seed_model`` is treated as a no-op alias for
        # ``--chatgpt_model``. The DFSDT backbone (which produces the
        # ancestor pseudo-tool) is selected by ``--chatgpt_model``; pass
        # ``--seed_model`` only if you want a warning when it disagrees.
        _refine_model = getattr(args, 'refine_model', None) or None
        _chat_model = getattr(args, 'chatgpt_model', None) or None
        _seed_model = getattr(args, 'seed_model', None) or None
        if _seed_model and _seed_model != _chat_model and process_id == 0:
            print(colored(
                f"[warn] --seed_model={_seed_model} differs from --chatgpt_model={_chat_model}; "
                "--seed_model is currently a documentation-only alias and is ignored — "
                "set --chatgpt_model to change the DFSDT backbone.",
                "yellow",
            ))
        if _refine_model and _refine_model != _chat_model:
            _refine_base_url = getattr(args, 'refine_base_url', None) or args.base_url
            self.refine_llm = build_chatgpt_function_llm(
                model=_refine_model,
                openai_key=args.openai_key,
                base_url=_refine_base_url,
            )
        else:
            # Sentinel: strategies fall back to the backbone ``llm`` parameter.
            self.refine_llm = None


        self.method = args.method
        self.input_query_file = args.input_query_file
        self.output_answer_file = args.output_answer_file
        self.query_id = query_id # passed from run_single_task
        self.tool_root_dir = args.tool_root_dir
        self.toolbench_key = args.toolbench_key
        self.rapidapi_key = args.rapidapi_key
        self.use_rapidapi_key = args.use_rapidapi_key
        self.api_customization = args.api_customization
        # Default to the local virtual API simulator (server/main.py serves POST /virtual).
        # Override with SERVICE_URL to point at a different simulator endpoint.
        self.service_url = os.getenv("SERVICE_URL", "http://localhost:8080/virtual")
        self.max_observation_length = args.max_observation_length
        self.observ_compress_method = args.observ_compress_method
        # Whether to refine tool descriptions in multi-turn retrieval
        self.retriever = retriever
        self.process_id = process_id
        # Counter for numbering sanity check outputs
        self.sanity_check_idx = 0

        self.tool_names = []
        self.cate_names = []

        self.input_description = query_json["query"] # this is the task query being run per query_id in the task list
        self.functions = []
        self.api_name_reflect = {}

        if self.just_query:
            data_dict = {"api_list": []}
            tool_descriptions = []
        elif self.retriever is not None:
            if args.retrieve_mode == 'normal':
                query_json, _ = retrieve_rapidapi_tools(self.retriever, self.input_description, args.retrieved_api_nums, args.tool_root_dir)
                data_dict = fetch_api_json(self.tool_root_dir, query_json)
                tool_descriptions = build_tool_description(self.tool_root_dir, data_dict)
            elif args.retrieve_mode == 'dynamic':
                data_dict = {"api_list":[]}
        else:
            data_dict = fetch_api_json(self.tool_root_dir, query_json)
            if len(data_dict["api_list"])!= len(tool_descriptions):
                tool_descriptions = build_tool_description(self.tool_root_dir, data_dict)

        for k,api_json in enumerate(data_dict["api_list"]):
            standard_tool_name = tool_descriptions[k][0]
            openai_function_json,cate_name, pure_api_name = api_json_to_openai_json(api_json,standard_tool_name)
            self.functions.append(openai_function_json)

            self.api_name_reflect[openai_function_json["function"]["name"]] = pure_api_name
            self.tool_names.append(standard_tool_name)
            self.cate_names.append(cate_name)

        finish_func = build_finish_schema()

        # 2026-05-25: in dynamic-retrieval mode, defer adding Finish until
        # after the first retrieval has actually happened. Otherwise
        # tool-eager reasoning models call Finish(give_up) at turn 0
        # without ever emitting the BEGIN/END description block, and
        # memetic never gets dispatched. The Finish tool is appended
        # post-retrieval inside dynamic_retrieve_base_on_des — see the
        # ``_finish_added`` flag there.
        self._finish_added = False
        self._finish_func = finish_func
        if not (hasattr(args, 'retrieve_mode') and args.retrieve_mode == 'dynamic'):
            self.functions.append(finish_func)
            self.tool_names.append(None)
            self.cate_names.append(None)
            self._finish_added = True
        self.CALL_MAX_TIME = 3
        if hasattr(args, 'retrieve_mode') and args.retrieve_mode == 'dynamic':
            self.task_description = dynamic_mode_preamble()
        else:
            unduplicated_reflection = {}
            for standardize_tool_name, tool_des in tool_descriptions:
                unduplicated_reflection[standardize_tool_name] = tool_des
            tool_lines = []
            for k, (standardize_tool_name, tool_des) in enumerate(unduplicated_reflection.items()):
                striped = (tool_des[:512].replace('\n','').strip() if isinstance(tool_des, str) else "") or "None"
                tool_lines.append(f"{k+1}.{standardize_tool_name}: {striped}")
            self.task_description = static_mode_preamble(tool_lines)

        self.success = 0

    def write_sanity_check(self, query_id, method, query, retrieval_iterations, instruction_file, retrieval_idx=0):
        """
        Writes a sanity check file for each retrieval initiation (which can have multiple retrievals inside if multi-turn):
        - query_id: ID of the query/task
        - method: retrieval method used
        - query: task query string
        - retrieval_iterations: list of retrieval steps for this initiation, ground truth APIs for this query, retrieved tools
            ground truth is included and we use sanity_check_idx in order to prevent overwriting
        - instruction_file: name of the instruction file (e.g., "G1_instruction")
        - retrieval_idx: integer index for this retrieval initiation for this task (Non-determinism)
        """
        base_dir = os.environ.get("FITTEXT_SANITY_DIR") or self.output_answer_file
        sanity_dir = os.path.join(base_dir, "sanity_check", str(query_id)) # two layers of directory structure for organizing retrieval initiations
        # make new directory to store the sanity check results
        os.makedirs(sanity_dir, exist_ok=True)
        instruction_base = os.path.splitext(os.path.basename(instruction_file))[0] # extract basename of absolute args.input_query_file
        out_path = os.path.join(
            sanity_dir,
            f"sanity_{query_id}_{method}_{instruction_base}_retrieval{retrieval_idx}.json"
        )
        # method and instruction_file information is stored in the path names
        output = {
            "query_id": query_id,
            "query": query,
            "retrieval_iterations": retrieval_iterations,
        }
        # load the original query file to fetch ground truth relevant APIs
        try:
            with open(self.input_query_file, "r") as reader:
                query_data = json.load(reader)
        except Exception:
            query_data = []

        ground_truth = []
        if isinstance(query_data, list):
            for entry in query_data:
                if entry.get("query_id") == query_id:
                    ground_truth = entry.get("relevant APIs", [])
                    break
        elif isinstance(query_data, dict):
            # sometimes file may be a mapping from ids to data dicts
            for entry in query_data.values():
                if isinstance(entry, dict) and entry.get("query_id") == query_id:
                    ground_truth = entry.get("relevant APIs", [])
                    break

        output["ground_truth"] = ground_truth

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    @staticmethod
    def _update_tool_memory_entries(tool_memory, retrieval_iterations):
        """
        Synchronize the shared tool memory with ancestor pseudo-tool intents.

        Only the normalized pseudo-tool strings are retained; retrieved tool
        payloads are intentionally omitted because tool_memory is used purely as
        an intent memory for duplication suppression.
        """
        if tool_memory is None:
            return

        # We iterate through the history of what actually happened
        for step in retrieval_iterations:
            # Extract the query (intent)
            pseudotool = step.get("current_description") or step.get("ancestor_description") or step.get("pseudotool")
            if pseudotool and str(pseudotool).strip():
                clean_key = " ".join(str(pseudotool).strip().split())

                # We only care about presence; values are placeholders for backward
                # compatibility with dict-based consumers.
                if isinstance(tool_memory, dict):
                    tool_memory[clean_key] = True

                # Fallback if tool_memory is still a set (backward compatibility)
                elif isinstance(tool_memory, set):
                    tool_memory.add(clean_key)

    def dynamic_retrieve_base_on_des(self, llm_output, retriever, llm, tool_memory=None):
        self.tool_memory = tool_memory if tool_memory is not None else {}
        self.retriever = retriever
        self.last_retrieval_message = None
        self.last_retrieval_details = None

        api_keys, retrieval_iterations, strategy_payload = select_and_run_strategy(llm_output, self, llm)
        strategy_name = (strategy_payload or {}).get("strategy") or "single_pass"
        if strategy_name not in {"memetic", "genetic", "memetic_toolret"}:
            self._update_tool_memory_entries(self.tool_memory, retrieval_iterations)

        def _format_desc_list(descs: List[str]) -> str:
            if not descs:
                return "• (no pseudo-tools were retained)"
            lines = []
            for idx, desc in enumerate(descs, start=1):
                snippet = (desc or "").strip()
                lines.append(f"  {idx}. {snippet}")
            return "\n".join(lines)

        def _format_winners(lineages: List[Dict[str, Any]]) -> str:
            if not lineages:
                return "• (no winning tools identified)"
            blocks: List[str] = []
            for idx, lineage in enumerate(lineages, start=1):
                ancestor = (lineage.get("ancestor") or "").strip()
                winners = lineage.get("winning_keys") or []
                if winners:
                    winner_text = ", ".join(
                        f"{w.get('tool_name', 'unknown')}.{w.get('api_name', 'unknown')}"
                        if w.get("tool_name") else w.get("api_name", "unknown")
                        for w in winners
                    )
                else:
                    winner_text = "no resolvable tools"
                if ancestor:
                    blocks.append(f"  Lineage {idx}: {ancestor} → {winner_text}")
                else:
                    blocks.append(f"  Lineage {idx}: {winner_text}")
            return "\n".join(blocks)

        message_body = ""

        if strategy_name == "scattershot":
            message_body = _format_winners(strategy_payload.get("lineages", []))
        else:
            message_body = _format_desc_list(strategy_payload.get("final_descriptions", []))

        header = "Another agent refined your pseudo-tools"
        if strategy_name:
            header += f" using {strategy_name.replace('_', ' ').upper()}"
        if message_body.strip():
            retrieval_message_text = f"{header}; your toolset is now:\n{message_body}"
        else:
            retrieval_message_text = f"{header}, but no pseudo-tools were retained."

        self.last_retrieval_message = {"role": "system", "content": retrieval_message_text}
        self.last_retrieval_details = {
            "strategy": strategy_name,
            "payload": strategy_payload,
            "retrieval_iterations": retrieval_iterations,
        }

        if not api_keys:
            self.retriever = None
            return retrieval_iterations

        # Keep only fields fetch_api_json expects; ignore lineage_index etc.
        api_list = [
            {"category_name": k["category_name"], "tool_name": k["tool_name"], "api_name": k["api_name"]}
            for k in api_keys
        ]
        query_json = {"api_list": api_list}  # wrap ONCE

        data_dict = fetch_api_json(self.tool_root_dir, query_json)
        tool_descriptions = build_tool_description(self.tool_root_dir, data_dict)

        for i, api_json in enumerate(data_dict["api_list"]):
            standard_tool_name = tool_descriptions[i][0]
            openai_function_json, cate_name, pure_api_name = api_json_to_openai_json(api_json, standard_tool_name)

            # dedupe by function name
            if openai_function_json["function"]["name"] not in self.api_name_reflect:
                self.api_name_reflect[openai_function_json["function"]["name"]] = pure_api_name
                self.functions.append(openai_function_json)
                self.tool_names.append(standard_tool_name)
                self.cate_names.append(cate_name)

        # 2026-05-25: append Finish to the toolset on the first successful
        # retrieval (deferred from wrapper init under dynamic mode — see
        # the constructor's ``_finish_added`` flag).
        if not getattr(self, "_finish_added", False):
            self.functions.append(getattr(self, "_finish_func", build_finish_schema()))
            self.tool_names.append(None)
            self.cate_names.append(None)
            self._finish_added = True

        self.retriever = None
        return retrieval_iterations
        
    @staticmethod
    def extract_relevant_api_descriptions_by_query_ids(json_data, target_query_ids):
        """
        This is not usually called unless it is for checking something.
        Used for organizational purposes.
        """
        target_set = set(target_query_ids)
        output = {}

        for entry in json_data:
            query_id = entry.get("query_id")
            if query_id not in target_set:
                continue

            relevant = set(tuple(pair) for pair in entry.get("relevant APIs", []))
            apis = entry.get("api_list", [])

            filtered = [
                (api["tool_name"], api["api_name"], api["api_description"])
                for api in apis
                if (api["tool_name"], api["api_name"]) in relevant
            ]

            output[query_id] = filtered

        return output

    def check_success(self):
        return self.success

    def to_json(self):
        return {}

    def restart(self):
        pass

    def get_score(self):
        return 0.0

    def step(self,**args):
        obs, code = self._step(**args)
        if len(obs) > self.max_observation_length:
            obs = obs[:self.max_observation_length] + "..."
        return obs, code

    def _step(self, action_name="", action_input=""):
        """Need to return an observation string and status code:
            0 means normal response
            1 means there is no corresponding api name
            2 means there is an error in the input
            3 represents the end of the generation and the final answer appears
            4 means that the model decides to pruning by itself
            5 represents api call timeout
            6 for 404
            7 means not subscribed
            8 represents unauthorized
            9 represents too many requests
            10 stands for rate limit
            11 message contains "error" field
            12 error sending request
        """
        if action_name == "Finish":
            try:
                json_data = json.loads(action_input,strict=False)
            except Exception:
                json_data = {}
                if '"return_type": "' in action_input:
                    if '"return_type": "give_answer"' in action_input:
                        return_type = "give_answer"
                    elif '"return_type": "give_up_and_restart"' in action_input:
                        return_type = "give_up_and_restart"
                    else:
                        return_type = action_input[action_input.find('"return_type": "')+len('"return_type": "'):action_input.find('",')]
                    json_data["return_type"] = return_type
                if '"final_answer": "' in action_input:
                    final_answer = action_input[action_input.find('"final_answer": "')+len('"final_answer": "'):]
                    json_data["final_answer"] = final_answer
            if "return_type" not in json_data.keys():
                return "{error:\"must have \"return_type\"\"}", 2
            if json_data["return_type"] == "give_up_and_restart":
                return "{\"response\":\"chose to give up and restart\"}",4
            elif json_data["return_type"] == "give_answer":
                if "final_answer" not in json_data.keys():
                    return "{error:\"must have \"final_answer\"\"}", 2
                
                self.success = 1 # succesfully return final_answer
                return "{\"response\":\"successfully giving the final answer.\"}", 3
            else:
                return "{error:\"\"return_type\" is not a valid choice\"}", 2
        else:
            for k, function_dict in enumerate(self.functions):
                function = function_dict['function']
                if function["name"].endswith(action_name):
                    pure_api_name = self.api_name_reflect[function["name"]]
                    payload = {
                        "category": self.cate_names[k],
                        "tool_name": self.tool_names[k],
                        "api_name": pure_api_name,
                        "tool_input": action_input,
                        "strip": self.observ_compress_method,
                        "toolbench_key": self.toolbench_key
                    }
                    if self.process_id == 0:
                        print(colored(f"query to {self.cate_names[k]}-->{self.tool_names[k]}-->{action_name}",color="yellow"))
                    if self.use_rapidapi_key or self.api_customization:
                        payload["rapidapi_key"] = self.rapidapi_key
                        response = get_rapidapi_response(payload, api_customization=self.api_customization)
                    else:
                        time.sleep(2) # rate limit: 30 per minute
                        headers = {"toolbench_key": self.toolbench_key}
                        timeout = None if self.service_url.endswith("virtual") else 15
                        # The single-process virtual tool server drops connections
                        # under concurrent load (requests.exceptions.ConnectionError:
                        # RemoteDisconnected). Previously this propagated up and
                        # crashed the whole trajectory (rc=1, losing the entire run).
                        # Treat a transient drop like the Timeout below: retry a few
                        # times with backoff, then soft-fail with status_code 5 so the
                        # agent gets an error observation and continues. Applies
                        # uniformly to every strategy, so it does not bias comparisons.
                        response = None
                        for _attempt in range(3):
                            try:
                                response = requests.post(self.service_url, json=payload, headers=headers, timeout=timeout)
                                break
                            except requests.exceptions.Timeout:
                                return json.dumps({"error": f"Timeout error...", "response": ""}), 5
                            except requests.exceptions.ConnectionError:
                                if _attempt == 2:
                                    return json.dumps({"error": f"Connection error...", "response": ""}), 5
                                time.sleep(3 * (_attempt + 1))
                            except requests.exceptions.RequestException:
                                # Any other request-layer failure (e.g.
                                # ChunkedEncodingError on a partial read) — same
                                # transient treatment: retry, then soft-fail.
                                if _attempt == 2:
                                    return json.dumps({"error": f"Request error...", "response": ""}), 5
                                time.sleep(3 * (_attempt + 1))
                        if response.status_code != 200:
                            return json.dumps({"error": f"request invalid, data error. status_code={response.status_code}", "response": ""}), 12
                        try:
                            response = response.json()
                        except Exception:
                            print(response)
                            return json.dumps({"error": f"request invalid, data error", "response": ""}), 12
                    # 1 Hallucinating function names
                    # 4 means that the model decides to pruning by itself
                    # 5 represents api call timeout
                    # 6 for 404
                    # 7 means not subscribed
                    # 8 represents unauthorized
                    # 9 represents too many requests
                    # 10 stands for rate limit
                    # 11 message contains "error" field
                    # 12 error sending request
                    if response["error"] == "API not working error...":
                        status_code = 6
                    elif response["error"] == "Unauthorized error...":
                        status_code = 7
                    elif response["error"] == "Unsubscribed error...":
                        status_code = 8
                    elif response["error"] == "Too many requests error...":
                        status_code = 9
                    elif response["error"] == "Rate limit per minute error...":
                        print("Reach api calling limit per minute, sleeping...")
                        time.sleep(10)
                        status_code = 10
                    elif response["error"] == "Message error...":
                        status_code = 11
                    else:
                        status_code = 0
                    return json.dumps(response), status_code
                    # except Exception as e:
                    #     return json.dumps({"error": f"Timeout error...{e}", "response": ""}), 5
            return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1


class pipeline_runner:
    def __init__(self, args, add_retrieval=False, process_id=0, server=False):
        self.args = args
        self.add_retrieval = add_retrieval
        self.process_id = process_id
        self.server = server
        
        if not self.server:
            self.task_list = self.generate_task_list()
        else:
            self.task_list = []

    def get_backbone_model(self):
        args = self.args
        if args.backbone_model == "toolllama":
            if not args.model_path:
                raise ValueError("Set --model_path or TOOLLLAMA_MODEL_PATH when using --backbone_model toolllama.")
            # ratio = 4 means the sequence length is expanded by 4, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
            ratio = int(args.max_sequence_length/args.max_source_sequence_length)
            replace_llama_with_condense(ratio=ratio)
            backbone_model = ToolLLaMA(model_name_or_path=args.model_path, max_sequence_length=args.max_sequence_length)
        else:
            backbone_model = args.backbone_model
        return backbone_model

    def get_retriever(self):
        server_url = getattr(self.args, "retriever_server_url", None)
        if server_url:
            # Derive corpus name (G1/G2/G3) from corpus_path
            corpus_name = next(
                (g for g in ("G1", "G2", "G3") if f"/{g}/" in self.args.corpus_path),
                "G1",
            )
            return RemoteToolRetriever(server_url=server_url, corpus_name=corpus_name)
        if getattr(self.args, "reinvoke", False):
            # Re-Invoke baseline (Chen et al. 2408.01875): synthetic-query-augmented
            # tool embeddings + intent-based multi-view ranking, as the STATIC
            # normal-mode retriever. The index must be pre-built
            # (baselines/build_reinvoke_stb_index.py); the adapter only loads it here.
            from toolbench.inference.LLM.reinvoke_retriever import ReInvokeSTBRetriever
            return ReInvokeSTBRetriever(
                corpus_path=self.args.corpus_path,
                model_path=self.args.retrieval_model_path,
                retrieved_api_nums=self.args.retrieved_api_nums,
                reinvoke_cfg={
                    "k_synth": int(os.environ.get("REINVOKE_K_SYNTH", "10")),
                    "max_intents": int(os.environ.get("REINVOKE_MAX_INTENTS", "3")),
                    # generator_model is resolved from configs/model_pins.yaml (or
                    # REINVOKE_GEN_MODEL) inside the adapter — never hardcode it here.
                    "cache_dir": os.environ.get("REINVOKE_CACHE_DIR"),
                    "index_build_concurrency": int(os.environ.get("REINVOKE_INDEX_CONCURRENCY", "8")),
                },
                # Solve path: require a pre-built index (no surprise build / no race).
                require_prebuilt=os.environ.get("REINVOKE_REQUIRE_PREBUILT", "1") == "1",
            )
        return ToolRetriever(corpus_path=self.args.corpus_path, model_path=self.args.retrieval_model_path, des_corpus=True)

    def get_args(self):
        return self.args

    def generate_task_list(self):
        args = self.args
        query_dir = args.input_query_file
        answer_dir = args.output_answer_file
        if not os.path.exists(answer_dir):
            os.makedirs(answer_dir)
        method = args.method
        backbone_model = self.get_backbone_model()
        white_list = get_white_list(args.tool_root_dir)
        task_list = []
        querys = json.load(open(query_dir, "r"))

        for query_id, data_dict in enumerate(querys):
            if "query_id" in data_dict:
                query_id = data_dict["query_id"]
            if "api_list" in data_dict:
                origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
                tool_des = contain(origin_tool_names,white_list)
                if tool_des == False:
                    continue
                tool_des = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]
            else:
                tool_des = None
            task_list.append((method, backbone_model, query_id, data_dict, args, answer_dir, tool_des))
        return task_list
    
    def method_converter(self, backbone_model, openai_key, method, env, process_id, single_chain_max_step=24, max_query_count=60, callbacks=None):
        if callbacks is None: callbacks = []
        if backbone_model == "chatgpt_function":
            llm_forward = ChatGPTFunction(model=self.args.chatgpt_model, openai_key=openai_key, base_url=self.args.base_url)
        else:
            model = backbone_model
            llm_forward = model
        
        if method.startswith("CoT"):
            passat = int(method.split("@")[-1])
            chain = single_chain(llm=llm_forward, io_func=env,process_id=process_id)
            result = chain.start(
                                pass_at=passat,
                                single_chain_max_step=single_chain_max_step,
                                answer=1)
        
        elif method.startswith("DFS"):
            pattern = r".+_w(\d+)"
            re_result = re.match(pattern,method)
            assert re_result != None
            width = int(re_result.group(1))
            with_filter = True
            if "woFilter" in method:
                with_filter = False
            # DFSDT
            # flow: pipeline_runner -> method_converter -> DFS_tree_search
            planning_llm = None
            no_planner = getattr(self.args, "no_planner", False)
            planning_model_name = getattr(self.args, "planning_model", None)
            if planning_model_name and not no_planner:
                if not openai_key:
                    if process_id == 0:
                        print("[warn] planning_model specified but openai_key missing; falling back to backbone model for planning.")
                else:
                    planning_llm = ChatGPTFunction(
                        model=planning_model_name,
                        openai_key=openai_key,
                        base_url=self.args.base_url,
                    )
            chain = DFS_tree_search(
                llm=llm_forward,
                io_func=env,
                process_id=process_id,
                callbacks=callbacks,
                planning_llm=planning_llm,
            )
            result = chain.start(
                                single_chain_max_step=single_chain_max_step,
                                tree_beam_size = width,
                                max_query_count = max_query_count,
                                answer=1,
                                with_filter=with_filter)
        else:
            print("invalid method")
            raise NotImplementedError
        return chain, result
    
    def run_single_task(self, method, backbone_model, query_id, data_dict, args, output_dir_path, tool_des, retriever=None, process_id=0, callbacks=None, server= None):
        if server is None:
            server = self.server
        if callbacks is None:
            if server: print("Warning: no callbacks are defined for server mode")
            callbacks = []
        splits = output_dir_path.split("/")
        os.makedirs("/".join(splits[:-1]),exist_ok=True)
        os.makedirs("/".join(splits),exist_ok=True)
        output_file_path = os.path.join(output_dir_path,f"{query_id}_{method}.json")
        
        # Uncomment for actual run
        if (not server) and os.path.exists(output_file_path):
            print(f"Skipping task {query_id} as output file already exists: {output_file_path}")
            return
        
        # Per-query LLM usage accounting: zero the process-local accumulator
        # before any LLM call for this query (one qid == one qa_pipeline
        # subprocess, so this is the whole query's budget). Snapshotted into
        # the result JSON below.
        reset_usage()

        [callback.on_tool_retrieval_start() for callback in callbacks]
        # environment created before instantiation of the decision tree at chain, result = self.method_converter

        env = rapidapi_wrapper(data_dict, tool_des, retriever, args, process_id=process_id, query_id=query_id)
        [callback.on_tool_retrieval_end(
            tools=env.functions
        ) for callback in callbacks]
        # query is only the query portion of the data_dict
        query = data_dict["query"]
        
        # data_dict is from task list, not from retrieval result (method_converter -> DFSDT -> retriever)
        # contains api_list, query, and relevant api from task_list

        if process_id == 0:
            print(colored(f"[process({process_id})] now playing: {query}, with {len(env.functions)} APIs", "green"))
        
        [callback.on_request_start(
            user_input=query,
            method=method,
        ) for callback in callbacks]


        # Budget caps. max_query_count is the OpenAI-query ceiling enforced
        # at DFS.py:351 (+5 slack → effective cap = max_query_count+5). Default
        # 20 (effective 25) is too tight for hard-subset queries that backtrack
        # heavily; pass --max_query_count via qa_pipeline_open_domain.py to
        # raise (e.g. 40-100) for budget-vulnerable cells.
        _max_qc = int(getattr(args, 'max_query_count', 20) or 20)
        _max_step = int(getattr(args, 'single_chain_max_step', 15) or 15)
        chain,result = self.method_converter(
            backbone_model=backbone_model, #backbone_model = chatgpt_function
            openai_key=args.openai_key,
            method=method,
            env=env,
            process_id=process_id,
            single_chain_max_step=_max_step,
            max_query_count=_max_qc,
            callbacks=callbacks
        )


        [callback.on_request_end(
            chain=chain.terminal_node[0].messages,
            outputs=chain.terminal_node[0].description,
        ) for callback in callbacks]
        if output_dir_path is not None:
            with open(output_file_path,"w") as writer:
                data = chain.to_json(answer=True,process=True)
                data["answer_generation"]["query"] = query
                # Per-query LLM call / token totals for efficiency reporting.
                # Keys: llm_calls, input_tokens, output_tokens, reasoning_tokens,
                # cached_input_tokens. is_solved is NOT here (it comes from the
                # judge); join post-hoc by query_id.
                data["answer_generation"]["usage_tracking"] = snapshot_usage()
                json.dump(data, writer, indent=2)
                success = data["answer_generation"]["valid_data"] and "give_answer" in data["answer_generation"]["final_answer"]
                print(colored(f"[process({process_id})]valid={success}", "green"))
        return result
        
    def run(self):
        task_list = self.task_list
        random.seed(42)
        random.shuffle(task_list)
        print(f"total tasks: {len(task_list)}")
        new_task_list = []
        for task in task_list:
            out_dir_path = task[-2]
            query_id = task[2]
            output_file_path = os.path.join(out_dir_path,f"{query_id}_{self.args.method}.json")
            if not os.path.exists(output_file_path):
                new_task_list.append(task)
        task_list = new_task_list
        print(f"undo tasks: {len(task_list)}")
        if self.add_retrieval:
            retriever = self.get_retriever()
        else:
            retriever = None
        for k, task in enumerate(task_list):
            print(f"process[{self.process_id}] doing task {k}/{len(task_list)}: real_task_id_{task[2]}")
            result = self.run_single_task(*task, retriever=retriever, process_id=self.process_id)
