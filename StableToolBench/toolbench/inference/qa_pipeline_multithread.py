'''
Close-domain QA Pipeline (multi-threaded)
'''

import argparse
import os
import sys
from pathlib import Path

# Load model pins so the CLI default is always sourced from the single source of truth.
_repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_repo_root))
from toolbench.observability.pins import load_pins as _load_pins
_DEFAULT_AGENT_MODEL = _load_pins(_repo_root)["agents"]["gpt_4_1_mini"]

from toolbench.inference.Downstream_tasks.rapidapi_multithread import pipeline_runner


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_model', type=str, default="chatgpt_function", required=False, help='chatgpt_function or davinci or toolllama')
    parser.add_argument('--chatgpt_model', type=str, default=_DEFAULT_AGENT_MODEL, required=False, help='Dated model tag. Default from configs/model_pins.yaml agents.gpt_4_1_mini.')
    parser.add_argument('--base_url', type=str, default="https://api.openai.com/v1", required=False, help='openai api url')
    parser.add_argument('--openai_key', type=str, default="", required=False, help='openai key for chatgpt_function or davinci model')
    parser.add_argument('--model_path', type=str, default=os.environ.get("TOOLLLAMA_MODEL_PATH", ""), required=False, help='')
    parser.add_argument('--tool_root_dir', type=str, default="", required=True, help='')
    parser.add_argument("--lora", action="store_true", help="Load lora model or not.")
    parser.add_argument('--lora_path', type=str, default=os.environ.get("TOOLLLAMA_LORA_PATH", ""), required=False, help='')
    parser.add_argument('--max_observation_length', type=int, default=1024, required=False, help='maximum observation length')
    parser.add_argument('--max_source_sequence_length', type=int, default=4096, required=False, help='original maximum model sequence length')
    parser.add_argument('--max_sequence_length', type=int, default=8192, required=False, help='maximum model sequence length')
    parser.add_argument('--single_chain_max_step', type=int, default=50, required=False, help='maximum step for single chain')
    parser.add_argument('--max_query_count', type=int, default=200, required=False, help='maximum query count')
    parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"], required=False, help='observation compress method')
    parser.add_argument('--method', type=str, default="CoT@1", required=False, help='method for answer generation: CoT@n,Reflexion@n,BFS,DFS,UCT_vote')
    parser.add_argument('--input_query_file', type=str, default="", required=False, help='input path')
    parser.add_argument('--output_answer_file', type=str, default="",required=False, help='output path')
    parser.add_argument('--toolbench_key', type=str, default="",required=False, help='your toolbench key to request rapidapi service')
    parser.add_argument('--rapidapi_key', type=str, default="",required=False, help='your rapidapi key to request rapidapi service')
    parser.add_argument('--use_rapidapi_key', action="store_true", help="To use customized rapidapi service or not.")
    parser.add_argument('--api_customization', action="store_true", help="To use customized api or not.")
    parser.add_argument('--num_thread', type=int, default=1, required=False, help='number of threads')
    parser.add_argument('--disable_tqdm', action="store_true", help="disable tqdm or not.")
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing runs')
    
    args = parser.parse_args()
    if args.overwrite:
        os.system(f"rm -rf {args.output_answer_file}")

    runner = pipeline_runner(args)
    runner.run()
