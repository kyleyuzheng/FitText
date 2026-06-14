"""
Resolution and standardization of tools retrieved during dynamic tool retrieval.
"""

# toolbench/inference/retrieval/resolver.py
import os, json
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from toolbench.utils import standardize, change_name
from toolbench.inference.LLM.functions import build_function_template, DESCRIPTION_MAX, TYPE_MAP

def get_white_list(tool_root_dir: str) -> Dict[str, Dict[str, str]]:
    white_list_dir = os.path.join(tool_root_dir)
    white_list: Dict[str, Dict[str, str]] = {}
    for cate in tqdm(os.listdir(white_list_dir)):
        p = os.path.join(white_list_dir, cate)
        if not os.path.isdir(p):
            continue
        for file in os.listdir(p):
            if not file.endswith(".json"):
                continue
            standard_tool_name = file.split(".")[0]
            with open(os.path.join(p, file)) as reader:
                js_data = json.load(reader)
            origin_tool_name = js_data["tool_name"]
            white_list[standardize(origin_tool_name)] = {
                "description": js_data["tool_description"],
                "standard_tool_name": standard_tool_name
            }
    return white_list

def contain(candidate_list: List[str], white_list: Dict[str, Any]):
    out = []
    for cand in candidate_list:
        if cand not in white_list:
            return False
        out.append(white_list[cand])
    return out

def fetch_api_json(tool_root_dir: str, query_json: Dict[str, Any]) -> Dict[str, Any]:
    data_dict = {"api_list": []}
    for item in query_json["api_list"]:
        cate_name = item["category_name"]
        tool_name = standardize(item["tool_name"])
        api_name = change_name(standardize(item["api_name"]))
        tool_json = json.load(open(os.path.join(tool_root_dir, cate_name, tool_name + ".json"), "r"))
        append_flag = False
        api_dict_names = []
        for api_dict in tool_json["api_list"]:
            api_dict_names.append(api_dict["name"])
            pure_api_name = change_name(standardize(api_dict["name"]))
            if pure_api_name != api_name:
                continue
            api_json = {}
            api_json["category_name"] = cate_name
            api_json["api_name"] = api_dict["name"]
            api_json["api_description"] = api_dict["description"]
            api_json["required_parameters"] = api_dict["required_parameters"]
            api_json["optional_parameters"] = api_dict["optional_parameters"]
            api_json["tool_name"] = tool_json["tool_name"]
            data_dict["api_list"].append(api_json)
            append_flag = True
            break
        if not append_flag:
            print(api_name, api_dict_names)
    return data_dict

"""def build_tool_description(self, data_dict):
        white_list = get_white_list(self.tool_root_dir)
        origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
        tool_des = contain(origin_tool_names,white_list)
        tool_descriptions = [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]
        return tool_descriptions"""

def build_tool_description(tool_root_dir: str, data_dict: Dict[str, Any]) -> List[List[str]]:
    white_list = get_white_list(tool_root_dir)
    origin_tool_names = [standardize(cont["tool_name"]) for cont in data_dict["api_list"]]
    tool_des = contain(origin_tool_names, white_list)
    if tool_des is False:
        missing = [name for name in origin_tool_names if name not in white_list]
        raise KeyError(
            f"Whitelist missing {len(missing)} tool(s): {missing}.\n"
            f"Query APIs: {[cont['tool_name'] for cont in data_dict['api_list']]}"
        )
    return [[cont["standard_tool_name"], cont["description"]] for cont in tool_des]

def api_json_to_openai_json(api_json: dict, standard_tool_name: str):
    """Updated from original ToolBench. The most up-to-date version of conversion to OpenAI function schema."""
    function_template = build_function_template()
    template = function_template["function"]

    # Name
    pure_api_name = change_name(standardize(api_json["api_name"]))
    template["name"] = (pure_api_name + f"_for_{standard_tool_name}")[-64:]

    # Description
    desc = f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
    api_desc = (api_json.get("api_description") or "").strip()
    if api_desc:
        truncated = api_desc.replace(api_json["api_name"], template["name"])[:DESCRIPTION_MAX]
        desc += f' The description of this function is: "{truncated}"'
    template["description"] = desc

    # Parameters
    props = template["parameters"]["properties"]
    req = template["parameters"]["required"]

    # Required parameters
    for para in api_json.get("required_parameters", []) or []:
        name = change_name(standardize(para["name"]))
        ptype = TYPE_MAP.get(para.get("type"), "string")
        prop = {
            "type": ptype,
            "description": (para.get("description") or "")[:DESCRIPTION_MAX],
        }
        default_value = para.get("default", "")
        if len(str(default_value)) != 0:
            prop["example_value"] = default_value
        props[name] = prop
        req.append(name)

    # Optional parameters (NO "optional" list in schema)
    for para in api_json.get("optional_parameters", []) or []:
        name = change_name(standardize(para["name"]))
        ptype = TYPE_MAP.get(para.get("type"), "string")
        prop = {
            "type": ptype,
            "description": (para.get("description") or "")[:DESCRIPTION_MAX],
        }
        default_value = para.get("default", "")
        if len(str(default_value)) != 0:
            prop["example_value"] = default_value
        props[name] = prop

    return function_template, api_json["category_name"], pure_api_name