import json


total_query_path = "path/to/instruction/G1_query.json"
filtered_query_path = "path/to/retrieval/data/G1_filtered_query.json"

dataset_tool_path = "path/to/test_instruction/G1_tool.json"
dataset_category_path = "path/to/test_instruction/G1_category.json"
dataset_instruction_path = "path/to/test_instruction/G1_instruction.json"

total_query = json.load(open(total_query_path, 'r'))

if dataset_tool_path:
    dataset_tool = json.load(open(dataset_tool_path, 'r'))
    tool_appeared = set()
    for data in dataset_tool:
        for api in data['api_list']:
            tool_appeared.add(api['tool_name'])
    print(f"Total Number of Tool appeared in the dataset: {len(tool_appeared)}")

    print(f"Total Number of query before filtering: {len(total_query)}")
    filtered_query = []
    for data in total_query:
        if not any(api['tool_name'] in tool_appeared for api in data['api_list']):
            filtered_query.append(data)
    
    total_query = filtered_query
    print(f"Total Number of query after filtering: {len(total_query)}")

if dataset_category_path:
    dataset_category = json.load(open(dataset_category_path, 'r'))
    category_appeared = set()
    for data in dataset_category:
        for api in data['api_list']:
            category_appeared.add(api['category_name'])
    print(f"Total Number of Category appeared in the dataset: {len(category_appeared)}")

    print(f"Total Number of query before filtering: {len(total_query)}")
    filtered_query = []
    for data in total_query:
        if not any(api['category_name'] in category_appeared for api in data['api_list']):
            filtered_query.append(data)
    
    total_query = filtered_query
    print(f"Total Number of query after filtering: {len(total_query)}")

if dataset_instruction_path:
    dataset_instruction = json.load(open(dataset_instruction_path, 'r'))
    instruction_appeared = set()
    for data in dataset_instruction:
        instruction_appeared.add(data['query'])
    print(f"Total Number of Instruction appeared in the dataset: {len(instruction_appeared)}")

    print(f"Total Number of query before filtering: {len(total_query)}")
    filtered_query = []
    for data in total_query:
        if isinstance(data['query'], list):
            data['query'] = data['query'][0]
        if data['query'] not in instruction_appeared:
            filtered_query.append(data)
    
    total_query = filtered_query
    print(f"Total Number of query after filtering: {len(total_query)}")

json.dump(total_query, open(filtered_query_path, 'w'), indent=4)



