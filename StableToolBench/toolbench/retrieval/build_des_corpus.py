import argparse
import pandas as pd
from tqdm import tqdm
import json
import os
from toolbench.utils import standardize, change_name
from toolbench.inference.LLM.chatgpt_function_model import ChatGPTFunction
from toolbench.prompt_template import FORMAT_TOOL_FUNCTIONARITY_FUNCTION, FORMAT_TOOL_FUNCTIONARITY_USER_FUNCTION

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_file', type=str, required=True, help='The path to the original corpus file')
parser.add_argument('--tool_root_dir', type=str, required=True, help='The root directory of the tools to be used for retrieval')
parser.add_argument('--gpt_model', type=str, required=True, help='The pinned model used for generated descriptions.')
parser.add_argument('--output_path', type=str, required=True, help='The path to the output file')
parser.add_argument('--refer_corpus_file', type=str, default="", nargs='+', required=True)
args = parser.parse_args()

corpus = pd.read_csv(args.corpus_file, sep='\t')

refer_corpus = []
for refer_file in args.refer_corpus_file:
    with open(refer_file, 'r', encoding='utf-8') as f:
        for line in f:
            tool_json = json.loads(line.strip())
            refer_corpus.append(tool_json)

start_index = 0
if os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0:
    last_line = None
    with open(args.output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                last_line = line.strip()
    print(f"Currently processing to the line: {last_line}")

    if last_line:
        try:
            last_record = json.loads(last_line)
            for idx, row in corpus.iterrows():
                doc = json.loads(row['document_content'])
                if (doc.get('category_name', '') == last_record['cate_name'] and
                    standardize(doc.get('tool_name', '')) == last_record['tool_name'] and
                    change_name(standardize(doc.get('api_name', ''))) == last_record['api_name']):
                    start_index = idx + 1
                    break
        except Exception as e:
            print(f"Having problem finding the last processing line in the corpus: {e}")
            start_index = 0

print(f"Having finished processing {start_index-1} records, start with processing {start_index} records.")

for row in tqdm(corpus.iloc[start_index:].itertuples(), total=len(corpus) - start_index):
    doc = json.loads(row.document_content)
    cate_name = doc.get('category_name', '')
    tool_name = standardize(doc.get('tool_name', ''))
    api_name = change_name(standardize(doc.get('api_name', '')))

    tool_json = json.load(open(os.path.join(args.tool_root_dir, cate_name, tool_name + ".json"), 'r'))
    api_json = {}
    api_doc = None
    for api_dict in tool_json["api_list"]:
        pure_api_name = change_name(standardize(api_dict["name"]))
        if pure_api_name != api_name:
            continue
        api_json["api_name"] = api_dict["name"]
        api_json["api_description"] = api_dict["description"]
        api_json["tool_name"] = tool_json["tool_name"]
        api_json["tool_description"] = tool_json["tool_description"]
        api_doc = str(api_dict)
        break

    if "api_name" not in api_json:
        print(f"API {api_name} not found in {tool_name}.json")
        continue

    for refer_api in refer_corpus:
        if refer_api["api_name"] == api_name and \
              refer_api["tool_name"] == tool_name and \
                refer_api["cate_name"] == cate_name:
            description = refer_api["functionality"]
            break
    else:
        llm = ChatGPTFunction(model=args.gpt_model, openai_key=os.getenv("OPENAI_API_KEY"))
        system = FORMAT_TOOL_FUNCTIONARITY_FUNCTION
        user = FORMAT_TOOL_FUNCTIONARITY_USER_FUNCTION
        llm.add_message({"role": "system", "content": system})
        user = user.replace("{api_name}", api_json["api_name"])
        user = user.replace("{api_description}", api_json["api_description"])
        user = user.replace("{tool_name}", api_json["tool_name"])
        user = user.replace("{tool_description}", str(api_json["tool_description"])) # Deal with the situation that tool_description may be none
        if len(api_doc) > 12000:
            api_doc = api_doc[:12000]
        user = user.replace("{api_doc}", api_doc)
        llm.add_message({"role": "user", "content": user})
        message, error_code, total_tokens = llm.parse(tools=None, process_id=None)
        description = message['content']

    data = {
        "functionality": description,
        "cate_name": cate_name,
        "tool_name": tool_name,
        "api_name": api_name,
    }

    with open(args.output_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')


