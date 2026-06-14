import argparse
import pandas as pd
from tqdm import tqdm
import json
import os
import sys
import time
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from typing import Optional

# Load model pins — single source of truth for all model identifiers.
_repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_repo_root / "StableToolBench"))
from toolbench.observability.pins import load_pins as _load_pins  # noqa: E402
_DEFAULT_CORPUS_MODEL: str = _load_pins(_repo_root)["agents"]["gpt_4_1_mini"]

SYSTEM_PROMPT = (
    "You are given the documentation about an API\n"
    "Your task: produce a clear, brief description of the API's functionality. Avoid including any unnecessary information unrelated to the API itself."
)

USER_PROMPT_TEMPLATE = (
    "API documentation:\n"
    "-----\n"
    "{doc}\n"
    "-----\n"
    "Please remember, tell me the functionality of this api without outputting unnecessary information, ONLY describing the API functionality in short description."
)

def normalize_sentence(s: str) -> str:
    s = (s or "").strip().replace("\n", " ").replace("  ", " ")
    if len(s) > 600:
        s = s[:600].rstrip(" ,;:") + "..."
    return s

def call_model(client: OpenAI, model: str, documentation: str, max_chars: int = 12000) -> str:
    doc = documentation[:max_chars] if documentation else ""
    user = USER_PROMPT_TEMPLATE.format(doc=doc)
    backoff = 2.0
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=512,
            )
            return normalize_sentence(resp.choices[0].message.content or "")
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff *= 1.8
    return ""


def read_last_processed_id(output_path: str) -> Optional[str]:
    """Return last processed id from output JSONL, if any."""
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return None
    last = None
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    try:
        obj = json.loads(last)
        return str(obj["id"])
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mangopy/ToolRet-Tools",
                        help="HF dataset repo_id, e.g. mangopy/ToolRet-Tools")
    parser.add_argument("--config", type=str, default="code",
                        choices=["code", "web", "customized"], help="Dataset config name (code/web/customized)")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to JSONL output (id, description). Appends and supports resume.")
    parser.add_argument("--model", type=str, default=_DEFAULT_CORPUS_MODEL,
                        help="OpenAI model for corpus description generation. Default from configs/model_pins.yaml agents.gpt_4_1_mini.")
    parser.add_argument("--start_from_scratch", action="store_true",
                        help="Ignore existing output and start from the first record.")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, args.config)['tools']

    last_id = None if args.start_from_scratch else read_last_processed_id(args.output_path)
    if last_id:
        print(f"[Resume] Last processed id in output: {last_id}", file=sys.stderr)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    processed = 0
    skipping = last_id is not None

    mode = "a" if os.path.exists(args.output_path) else "w"

    with open(args.output_path, mode, encoding="utf-8") as out_f:
        for row in tqdm(ds, total=len(ds), desc="Processing"):
            tid = str(row.get("id", ""))
            doc = row.get("documentation", "")

            if skipping:
                if tid == last_id:
                    skipping = False
                continue

            if not tid:
                continue
            if not doc:
                description = "Provides functionality as described in its documentation."
            else:
                description = call_model(client, args.model, doc)

            record = {"id": tid, "description": description}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            processed += 1

    print(f"Done. Wrote {processed} descriptions to {args.output_path}")    

if __name__ == "__main__":
    main()