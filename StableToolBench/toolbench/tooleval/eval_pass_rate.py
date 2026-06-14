from evaluators import load_registered_automatic_evaluator
import os
import json
import csv
from evaluators.registered_cls.rtl import AnswerStatus, TaskStatus, AnswerPass
import random
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from tqdm import tqdm
import numpy as np
from utils import test_sets, get_steps
import backoff

abs_dir = os.path.split(__file__)[0]

def default_evaluator_name() -> str:
    return os.environ.get("STB_EVALUATOR", "tooleval_judge_default")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--converted_answer_path', type=str, default="", required=True, help='converted answer path')
    parser.add_argument('--save_path', type=str, default="", required=False, help='result save path')
    parser.add_argument('--reference_model', type=str, default="", required=False, help='model predictions path')
    parser.add_argument('--test_ids', type=str, default="", required=True, help='model predictions path')
    parser.add_argument('--evaluator', type=str, default=default_evaluator_name(), required=False, help='which evaluator to use.')
    parser.add_argument('--max_eval_threads', type=int, default=30, required=False, help='max threads nums')
    parser.add_argument('--evaluate_times', type=int, default=4, required=False, help='how many times to predict with the evaluator for each solution path.')
    parser.add_argument('--test_set', nargs='+', default=['G1_instruction'], help='test set name')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite the existing result file')
    parser.add_argument(
        '--missing_prediction_policy',
        choices=['zero', 'error', 'ignore'],
        default='zero',
        help=(
            "How to handle solvable qids absent from converted predictions. "
            "'zero' materializes them as Unsolved so the denominator stays the "
            "full solvable subset; 'error' aborts; 'ignore' preserves legacy "
            "len(label_cnt) behavior."
        ),
    )
    parser.add_argument(
        '--judge_details_first',
        action='store_true',
        help=(
            "Audit mode: ask the evaluator to inspect the full converted "
            "trajectory before issuing a solve label. Default preserves "
            "StableToolBench's legacy final-answer-first official behavior."
        ),
    )
    parser.add_argument(
        '--save_reasons',
        action='store_true',
        help='Store evaluator reasons in the output JSON under is_solved_reason.',
    )
    return parser.parse_args()

def write_results(filename: str, reference_model: str, label_cnt: dict) -> None:
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["query", "available_tools", "model_intermediate_steps", "model_final_step", "model", "query_id", "is_solved", ])
        for query_id in label_cnt:
            query = label_cnt[query_id]["query"]
            tool_names = label_cnt[query_id]["tool_names"]
            answer_steps = label_cnt[query_id]["answer_steps"]
            final_step = label_cnt[query_id]["final_step"]
            is_solved = label_cnt[query_id]["is_solved"]
            writer.writerow([query, tool_names, answer_steps, final_step, reference_model, query_id, is_solved,])

def load_test_instruction(test_ids_root: str, test_set: str) -> dict:
    instruction_path = os.path.join(os.path.dirname(test_ids_root), "test_instruction", test_set + ".json")
    if not os.path.exists(instruction_path):
        return {}
    with open(instruction_path, "r") as f:
        rows = json.load(f)
    return {str(row.get("query_id")): row for row in rows if "query_id" in row}

def tool_names_from_instruction(example: dict) -> list:
    names = []
    for tool in example.get("api_list", []):
        api_name = str(tool.get("api_name", "")).strip().replace(" ", "_").lower()
        tool_name = str(tool.get("tool_name", "")).strip().replace(" ", "_").lower()
        if api_name and tool_name:
            names.append(f"{api_name}_for_{tool_name}")
    return names

def materialize_missing_label(
    label_cnt: dict,
    query_id: str,
    instruction_examples: dict,
    evaluate_times: int,
) -> None:
    example = instruction_examples.get(str(query_id), {})
    label_cnt[str(query_id)] = {
        "query": example.get("query", ""),
        "tool_names": tool_names_from_instruction(example),
        "answer_steps": "",
        "final_step": "MISSING_PREDICTION",
        "is_solved": {str(i): str(AnswerStatus.Unsolved) for i in range(evaluate_times)},
        "missing_prediction": True,
    }
            

if __name__ == "__main__":
    args = parse_args()
    if not args.evaluator:
        raise RuntimeError(
            "No evaluator configured. Pass --evaluator, set STB_EVALUATOR, "
            "or use the bundled tooleval_judge_default evaluator."
        )
    detail_first = args.judge_details_first or args.evaluator.endswith("_strict")
    if detail_first and not args.judge_details_first:
        print(f"Evaluator {args.evaluator} forces trajectory-first judging.", flush=True)
    evaluators = [load_registered_automatic_evaluator(evaluator_name=args.evaluator, evaluators_cfg_path=os.path.join(abs_dir,'evaluators')) for _ in range(args.max_eval_threads)]
    
    @backoff.on_exception(backoff.expo, Exception, max_time=15)
    def compute_pass_rate(query_id, example, evaluate_time):
        global evaluators
        evaluator = random.choice(evaluators)
        answer_steps, final_step = get_steps(example)
        
        if "'name': 'Finish'" not in final_step:
            return query_id, AnswerStatus.Unsolved, evaluate_time, "Missing Finish final step"
        
        is_solved, is_solved_reason = evaluator.check_is_solved(
            {
                'query':example['query'],
                'available_tools':example['available_tools'],
            },
            example['answer'],
            return_reason=True,
            detail_first=detail_first,
        )

        # return query_id, task_solvable, is_solved, label, reason, not_hallucinate, evaluate_time
        return query_id, is_solved, evaluate_time, is_solved_reason
        
    reference_model = args.reference_model
    output_list = []
    for test_set in args.test_set:
        reference_path = f"{args.converted_answer_path}/{reference_model}/{test_set}.json"
        test_ids = [str(qid) for qid in json.load(open(os.path.join(args.test_ids, test_set+".json"), "r")).keys()]
        instruction_examples = load_test_instruction(args.test_ids, test_set)
        reference_examples = json.load(open(reference_path, "r"))
        if os.path.exists(f"{args.save_path}/{test_set}_{reference_model}.json") and not args.overwrite:
            existed_ids = list(json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r")).keys())
            label_cnt = json.load(open(f"{args.save_path}/{test_set}_{reference_model}.json", "r"))
        else:
            existed_ids = []
            label_cnt = {}
        
        with ThreadPoolExecutor(args.max_eval_threads) as pool:
            future = []
            for query_id in reference_examples:
                if str(query_id) not in test_ids:
                    continue
                if query_id in existed_ids:
                    continue
                for i in range(args.evaluate_times):
                    example = reference_examples[query_id]
                    future.append(pool.submit(
                        compute_pass_rate,
                        query_id,
                        example,
                        evaluate_time=i
                    ))

            for thd in tqdm(as_completed(future),total=len(future),ncols=100):
                query_id, is_solved, evaluate_time, is_solved_reason = thd.result()
                example = reference_examples[query_id]
                query = example["query"]
                tool_names = []
                for tool_dict in example["available_tools"]:
                    if 'function' in tool_dict:
                        tool_name = tool_dict["function"]['name']
                    else:
                        tool_name = tool_dict["name"]
                    tool_names.append(tool_name)
                answer_steps, final_step = get_steps(example)
                if query_id not in label_cnt:
                    label_cnt[query_id] = {}
                label_cnt[query_id]["query"] = query
                label_cnt[query_id]["tool_names"] = tool_names
                label_cnt[query_id]["answer_steps"] = answer_steps
                label_cnt[query_id]["final_step"] = final_step
                if 'is_solved' not in label_cnt[query_id]:
                    label_cnt[query_id]["is_solved"] = {}
                label_cnt[query_id]["is_solved"][evaluate_time] = str(is_solved)
                if args.save_reasons:
                    if 'is_solved_reason' not in label_cnt[query_id]:
                        label_cnt[query_id]["is_solved_reason"] = {}
                    label_cnt[query_id]["is_solved_reason"][evaluate_time] = is_solved_reason
                json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)
        missing_ids = [query_id for query_id in test_ids if query_id not in {str(k) for k in label_cnt}]
        if missing_ids and args.missing_prediction_policy == "error":
            raise RuntimeError(
                f"{test_set}: {len(missing_ids)} solvable qids missing from converted predictions: "
                + ", ".join(missing_ids[:20])
            )
        if missing_ids and args.missing_prediction_policy == "zero":
            for query_id in missing_ids:
                materialize_missing_label(label_cnt, query_id, instruction_examples, args.evaluate_times)
        json.dump(label_cnt, open(f"{args.save_path}/{test_set}_{reference_model}.json", "w"), ensure_ascii=False, indent=4)
        
        filename = f"{args.save_path}/{test_set}_{reference_model}.csv"
        write_results(filename, reference_model, label_cnt)
        scores = []
        if args.missing_prediction_policy == "ignore":
            score_query_ids = [str(query_id) for query_id in label_cnt]
        else:
            score_query_ids = test_ids
        for runtime in range(args.evaluate_times):
            score = 0 
            for query_id in score_query_ids:
                solved_dict = {**label_cnt[query_id]['is_solved']}
                solved_dict = {int(k):v for k,v in solved_dict.items()}
                if runtime not in solved_dict:
                    score += 0
                    continue
                if solved_dict[runtime] == "AnswerStatus.Solved":
                    score += 1
                elif solved_dict[runtime] == "AnswerStatus.Unsure":
                    score += 0.5

            
            scores.append(score / len(score_query_ids))
    

        # Create a file to store the results
        result_path = os.path.join(args.save_path, f"{reference_model}_pass_rate.txt")

        print(f"Test set: {test_set}. Model: {reference_model}.")
        solve_rate = sum(scores) / len(scores) * 100
        std_dev = np.std(scores).item() * 100
        print(f"Solve rate: {solve_rate:.1f}% Std: {std_dev:.1f}%")

        with open(result_path, 'a') as f:
            f.write(f"Test set: {test_set}. Model: {reference_model}.\n")
            f.write(f"Solve rate: {solve_rate:.1f}% Std: {std_dev:.1f}%\n")      

        
