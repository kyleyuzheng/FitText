
import json
import argparse
import os
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="", required=True, help='The directory to output the split files')
parser.add_argument('--train_query_file', type=str, default="", required=True, help='The name of the query file')
parser.add_argument('--test_query_file', type=str, default="", nargs='+', required=True, help='The name of the index file')

args = parser.parse_args()

with open(args.train_query_file, 'r') as f:
    query_train = json.load(f)

query_test = []
for test_file in args.test_query_file:
    with open(test_file, 'r') as f:
        query_test += json.load(f)

os.makedirs(args.output_dir, exist_ok=True)
train_file = f"{args.output_dir}/train.json"
test_file = f"{args.output_dir}/test.json"

with open(train_file, 'w') as f:
    json.dump(query_train, f)

with open(test_file, 'w') as f:
    json.dump(query_test, f)


### For dataset preprocess ###

doc_id_map = {}  # Create a mapping from doc to doc_id
query_id_map = {}  # Create a mapping from query to query_id

documents = []
train_pairs = []
test_pairs = []

def process_data(data, pairs):
    for doc in tqdm(data):
        for api in doc['api_list']:
            document_content = api
            api_identity = [api['tool_name'], api['api_name']]

            document_content_str = json.dumps(document_content)

            if document_content_str not in doc_id_map:
                doc_id = len(doc_id_map) + 1 
                doc_id_map[document_content_str] = doc_id
                documents.append([doc_id, document_content_str])
            else:
                doc_id = doc_id_map[document_content_str]

            # Check if the current API is in the relevant APIs
            if api_identity in doc['relevant APIs']:
                query = doc['query']
                if isinstance(query, list):
                    query = query[0] # a few instances is store in list
                query_id = query_id_map.setdefault(query, len(query_id_map) + 1)
                pairs.append(([query_id, query], [query_id, 0, doc_id, 1]))

process_data(query_train, train_pairs)
process_data(query_test, test_pairs)
            
# Shuffle the data using the shuffle function
train_pairs = shuffle(train_pairs, random_state=42)
test_pairs = shuffle(test_pairs, random_state=42)

# Split the shuffled data into queries and labels
train_queries, train_labels = zip(*train_pairs)
test_queries, test_labels = zip(*test_pairs)

documents_df = pd.DataFrame(documents, columns=['docid', 'document_content'])
train_queries_df = pd.DataFrame(train_queries, columns=['qid', 'query_text'])
train_labels_df = pd.DataFrame(train_labels, columns=['qid', 'useless', 'docid', 'label'])
test_queries_df = pd.DataFrame(test_queries, columns=['qid', 'query_text'])
test_labels_df = pd.DataFrame(test_labels, columns=['qid', 'useless', 'docid', 'label'])



# Save as .tsv and .txt files
documents_df.to_csv(args.output_dir + '/raw_corpus.tsv', sep='\t', index=False)
train_queries_df.to_csv(args.output_dir + '/train.query.txt', sep='\t', index=False, header=False)
test_queries_df.to_csv(args.output_dir + '/test.query.txt', sep='\t', index=False, header=False)
train_labels_df.to_csv(args.output_dir + '/qrels.train.tsv', sep='\t', index=False, header=False)
test_labels_df.to_csv(args.output_dir + '/qrels.test.tsv', sep='\t', index=False, header=False)