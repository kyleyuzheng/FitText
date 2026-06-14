"""Minimal import smoke test -- run after setup to verify installation."""
import sys

def main():
    errors = []
    modules = [
        ("toolbench.inference.Downstream_tasks.rapidapi", "pipeline_runner"),
        ("toolbench.inference.Downstream_tasks.strategies", "select_and_run_strategy"),
        ("toolbench.inference.Algorithms.DFS", "DFS_tree_search"),
        ("toolbench.inference.LLM.chatgpt_function_model", "ChatGPTFunction"),
        ("toolbench.inference.LLM.retriever", "ToolRetriever"),
        ("toolbench.retrieval.services", "retrieve_rapidapi_tools"),
        ("toolbench.retrieval.resolver", "get_white_list"),
    ]
    for mod_name, attr in modules:
        try:
            mod = __import__(mod_name, fromlist=[attr] if attr else [])
            if attr:
                getattr(mod, attr)
            print(f"  OK  {mod_name}")
        except Exception as e:
            print(f"  FAIL {mod_name}: {e}")
            errors.append(mod_name)

    if errors:
        print(f"\n{len(errors)} module(s) failed to import.")
        sys.exit(1)
    else:
        print("\nAll core imports successful.")
        sys.exit(0)

if __name__ == "__main__":
    main()
