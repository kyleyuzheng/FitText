import argparse
import os
from datasets import load_dataset
from eval_toolret import dataset_categories, cal_eval, add_run_results
from tqdm import tqdm
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="dbd", choices=["dbd", "scattershot", "single_pass", "just_query"])
    parser.add_argument("--dataset", type=str, default="code", choices=["code", "web", "customized"],
                        help="Dataset config name (code/web/customized)")
    parser.add_argument("--max_turns_dbd", type=int, default=3, help="Number of DBD refinement turns (if using DBD strategy)")
    parser.add_argument("--retrieved_api_nums", type=int, default=5)
    parser.add_argument("--result_file_path", type=str, default="./result")
    parser.add_argument("--output_path", type=str, default="./result")
    parser.add_argument(
        "--score_mode",
        choices=["rank", "cosine_legacy"],
        default="rank",
        help=(
            "Use strictly decreasing rank scores by default so pytrec_eval honors "
            "the stored retrieval order. This keeps the same NDCG metric and only "
            "changes run-file score projection. cosine_legacy reproduces older "
            "raw-score reranking."
        ),
    )

    args = parser.parse_args()

    result_file = os.path.join(args.result_file_path, args.dataset, args.strategy)
    output_path = os.path.join(args.output_path, args.dataset, args.strategy, "overall_results_for_each_turn.txt")

    with open(output_path, "w") as f:
        f.write("=== Start Evaluation ===\n")
        f.write(f"score_mode={args.score_mode}\n")

    for i in range(1, args.max_turns_dbd + 1):
        is_final_turn = False
        if i == args.max_turns_dbd:
            is_final_turn = True
            i = "final"
        qrels_all, results_all = {}, {}

        per_subset_metrics = {}
        per_subset_qcounts = {}
        with open(output_path, "a") as f:
            f.write(f"=== Per-subset metrics for turn {i} ===\n")

        for subset_name in dataset_categories[args.dataset]:
            print(f"Processing subset: {subset_name}")

            with open(os.path.join(result_file, f"{subset_name}.jsonl"), "r") as f:
                subset_result_data = [json.loads(line) for line in f.readlines()]

            sub_dataset = load_dataset("mangopy/ToolRet-Queries", subset_name)['queries']
            qrels = {}
            results = {}
            for idx, sample in enumerate(tqdm(sub_dataset, desc="Proccessing samples")):
                qid = sample['id']
                gt_tools = json.loads(sample['labels'])
                qrels[qid] = {str(x['id']): int(x['relevance']) for x in gt_tools}

                retrieved_tools = []
                retrieved_tool_scores = []
                if args.strategy == "dbd":
                    data_list = subset_result_data[idx]
                    for data in data_list:
                        # Check whether the data have the key turn
                        if str(i) in data:
                            retrieved_tools.extend(data[str(i)]['retrieved_tools'][:args.retrieved_api_nums])
                            retrieved_tool_scores.extend(data[str(i)]['retrieved_tools_scores'][:args.retrieved_api_nums])
                        elif is_final_turn and "iteration" in data and data["iteration"] == "final":
                            retrieved_tools.extend(data['retrieved_tools'][:args.retrieved_api_nums])
                            retrieved_tool_scores.extend(data['retrieved_tools_scores'][:args.retrieved_api_nums])

                results[qid] = {}
                add_run_results(results[qid], retrieved_tools, retrieved_tool_scores, args.score_mode)
            
            subset_metrics, qcount = cal_eval(qrels, results)
            per_subset_metrics[subset_name] = subset_metrics
            per_subset_qcounts[subset_name] = qcount

            line = f"{subset_name} {subset_metrics} (#queries={qcount})\n"
            with open(output_path, "a", encoding="utf-8") as f:
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a") as f:
            f.write(f"\n=== Global metrics for turn {i} over all evaluated queries ===\n")
            f.write(f"{global_metrics} (#queries={global_qcount})\n")

            f.write(f"\n=== Weighted-avg over subsets for turn {i} (by #queries, optional) ===\n")
            f.write(f"{weighted_metrics}\n")

if __name__ == "__main__":
    main()
