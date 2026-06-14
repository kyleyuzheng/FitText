'''
Open-domain QA Pipeline
'''
import argparse
import os
import sys
from pathlib import Path

# Load model pins so CLI defaults are always sourced from the single source of truth.
_repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_repo_root))
from toolbench.observability.pins import load_pins as _load_pins
_pins = _load_pins(_repo_root)
_DEFAULT_AGENT_MODEL = _pins["agents"]["gpt_4_1_mini"]
_DEFAULT_PLANNING_MODEL = _pins["agents"]["o3_mini"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieve_mode', type=str, default="normal", choices=["normal", "dynamic"], required=False, help='normal or dynamic')
    parser.add_argument('--corpus_path', type=str, default="", required=False, help='')
    parser.add_argument('--retrieval_model_path', type=str, default="", required=False, help='')
    parser.add_argument('--retriever_server_url', type=str, default=None, required=False, help='URL of shared retriever server (e.g. http://localhost:8090). When set, skips local SimCSE loading.')
    parser.add_argument('--reinvoke', action='store_true', default=False, help='Use the Re-Invoke retriever (synthetic-query-augmented embeddings + intent-based multi-view ranking) as the static normal-mode retriever. Pre-build the index with baselines/build_reinvoke_stb_index.py. See toolbench/inference/LLM/reinvoke_retriever.py.')
    parser.add_argument('--retrieved_api_nums', type=int, default=5, required=False, help='Final number of retrieved API per retrieval added to functions list')
    parser.add_argument('--backbone_model', type=str, default="chatgpt_function", required=False, help='chatgpt_function or toolllama')
    parser.add_argument('--chatgpt_model', type=str, default=_DEFAULT_AGENT_MODEL, required=False, help='Dated model tag. Default from configs/model_pins.yaml agents.gpt_4_1_mini.')
    parser.add_argument('--base_url', type=str, default="https://api.openai.com/v1", required=False, help='openai api url')
    parser.add_argument('--openai_key', type=str, default="", required=False, help='openai key for chatgpt_function model')
    parser.add_argument('--model_path', type=str, default=os.environ.get("TOOLLLAMA_MODEL_PATH", ""), required=False, help='')
    parser.add_argument('--tool_root_dir', type=str, default="", required=True, help='')
    parser.add_argument('--max_observation_length', type=int, default=1024, required=False, help='maximum observation length')
    parser.add_argument('--max_source_sequence_length', type=int, default=4096, required=False, help='original maximum model sequence length')
    parser.add_argument('--max_sequence_length', type=int, default=8192, required=False, help='maximum model sequence length')
    parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"], required=False, help='maximum observation length')
    parser.add_argument('--method', type=str, default="CoT@1", required=False, help='method for answer generation: CoT@n,Reflexion@n,BFS,DFS,UCT_vote')
    parser.add_argument('--input_query_file', type=str, default="", required=False, help='input path')
    parser.add_argument('--output_answer_file', type=str, default="",required=False, help='output path')
    parser.add_argument('--toolbench_key', type=str, default="",required=False, help='your toolbench key to request rapidapi service')
    parser.add_argument('--rapidapi_key', type=str, default="",required=False, help='your rapidapi key to request rapidapi service')
    parser.add_argument('--use_rapidapi_key', action="store_true", help="To use customized rapidapi service or not.")
    parser.add_argument('--api_customization', action="store_true", help="To use customized api or not. NOT SUPPORTED currently under open domain setting.")
    
    # ----------------------------------------------------------------------------
    
    parser.add_argument('--planning_model',type=str,default=_DEFAULT_PLANNING_MODEL,required=False,help="Default reasoning model for root planner. Default from configs/model_pins.yaml agents.o3_mini.") # root planner DFSDT for dynamic and normal modes
    parser.add_argument('--no_planner', action='store_true', help='Disable the root planner gate entirely.')
    parser.add_argument('--dbd', action='store_true',help='Enable description-by-description (lineage-local) dynamic retrieval.')

    parser.add_argument('--refinement', action='store_true',help='Refine (True) vs Re-generate (False). Default False => regeneration.')
    parser.add_argument('--dbd-refine-turns', type=int, default=0, help='Number of DBD (refinment and regen.) turns per lineage for DBD.')

    parser.add_argument('--disable-midexec-retrieval', action='store_true',
                        help='Root-node ablation: gate the DFS mid-execution retrieval trigger '
                             '(suppress re-retrieval when the agent emits <|begin_func_description|> '
                             'during execution).')
    parser.add_argument('--refine_style', type=str, default='fittext', choices=['fittext', 'xu'],
                        help="Prompt style for the dbd refine LLM call. 'fittext' (default) refines a "
                             "pseudo-tool description (FitText axis-a). 'xu' refines the user instruction "
                             "(Xu et al. 2024 method-class axis-a). Used only when --refinement is true.")

    parser.add_argument('--scattershot', action='store_true', help='Enables generation with scattershot and size is for number of samples.')
    parser.add_argument('--size', type=int, default=5, help="Number of samples for scattershot.")
    parser.add_argument('--just_query', action='store_true',
                        help='Zero-retrieval parametric floor baseline: answer from the query alone '
                             '(no tool retrieval, no tool calls). Routed FIRST in '
                             'strategies.select_and_run_strategy; short-circuits the planner and every '
                             'other strategy. Pair with --retrieve_mode normal.')

     # ----------------------------------------------------------------------------
     
    parser.add_argument('--genetic', action='store_true', help='Enable genetic pseudo-tool evolution.')
    parser.add_argument('--memetic', action='store_true', help='Enable memetic search (LLM refinement + genetic).')
    parser.add_argument('--population_size', type=int, default=5, help='Population size for genetic/memetic search.')
    parser.add_argument('--generation_num', type=int, default=3, help='Number of generations for genetic/memetic search.')
    parser.add_argument('--similarity_threshold', type=float, default=0.95, help='Early-stop threshold on retrieval similarity.')
    parser.add_argument('--memetic_toolret', action='store_true', help='Enable memetic with ToolRet retrieval seeding.')
    parser.add_argument('--base_temp', type=float, default=0.9, help='Base temperature for strategy LLM calls (memetic/scattershot).')
    parser.add_argument('--evolution_temperature', type=float, default=None,
                        help='Memetic evolution-subprocess rescue T (seed/mutation/crossover/refine). '
                             'None → fall back to base_temp. Recommended split-T setting: base_temp=1.0, evolution_temperature=1.5.')
    parser.add_argument('--max_query_count', type=int, default=20,
                        help='DFSDT outer-loop LLM-call budget per cell. Effective cap = this+5 (DFS.py:351). '
                             'Default 20 (effective 25) matches the paper baseline. Bump to 40-100 for '
                             'budget-vulnerable hard cells.')
    parser.add_argument('--single_chain_max_step', type=int, default=15,
                        help='Max DFSDT tree depth per chain. Default 15 matches paper baseline.')
    parser.add_argument('--seed_model', type=str, default=None, help='Model for seeding population (dual-LLM).')
    parser.add_argument('--refine_model', type=str, default=None, help='Model for refinement (dual-LLM).')
    parser.add_argument('--seed_base_url', type=str, default=None, help='API base URL for seed model (vLLM).')
    parser.add_argument('--refine_base_url', type=str, default=None, help='API base URL for refine model (vLLM).')
    # launch.json configurations
    args = parser.parse_args()
    if not args.openai_key:
        args.openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not args.toolbench_key:
        args.toolbench_key = os.environ.get("TOOLBENCH_KEY", "")
    if args.dbd and args.retrieve_mode != 'dynamic':
        raise ValueError('DBD requires --retrieve-mode dynamic.')
    if args.reinvoke and args.retrieve_mode != 'normal':
        raise ValueError('--reinvoke requires --retrieve_mode normal (Re-Invoke is a pure static retriever; it never re-retrieves mid-trajectory).')
    if args.scattershot and args.dbd:
        raise ValueError('Scattershot is incompatible with DBD.')
    if args.refinement and not args.dbd:
        # The converse is allowed: if dBd is set but refinement is not, we do re-generation.
        raise ValueError("Refinement flag requires dBd to be set.")
    from toolbench.inference.Downstream_tasks.rapidapi import pipeline_runner

    runner = pipeline_runner(args, add_retrieval=True)
    runner.run()
    # use launch.json debugger with this file open or maybe bugs
