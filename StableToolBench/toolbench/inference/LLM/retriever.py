import time
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json
import re
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from toolbench.utils import standardize, standardize_category, change_name, process_retrieval_ducoment, process_des_retrieval_document
from transformers import AutoTokenizer, AutoModel

class ToolRetriever:
    def __init__(self, corpus_path = "", model_path="", des_corpus=False):
        # Processing the CLI arguments
        self.corpus_path = corpus_path
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.des_corpus = des_corpus

        # Data Loading and Processing Step
        self.corpus, self.corpus2tool = self.build_retrieval_corpus()
        # corpus is the descriptions from des_corpus.json per G1, G2, and G3
        # corpus2tool is a dict mapping the description to the category, tool name, and api name
        # e.g. {'description': 'category_name[SEP]tool_name[SEP]api_name'}
        

        self.build_retrieval_embedder()
        self.corpus_embeddings = self.build_corpus_embeddings() # sets to the CLI arg by calling build_corpus_embeddings des_corpus.json
        
    def build_retrieval_corpus(self):
        print("Building corpus...")
        if self.des_corpus:
            corpus, corpus2tool = process_des_retrieval_document(self.corpus_path) # des_corpus.json
        else:
            documents_df = pd.read_csv(self.corpus_path, sep='\t')
            corpus, corpus2tool = process_retrieval_ducoment(documents_df)
            corpus_ids = list(corpus.keys())
            corpus = [corpus[cid] for cid in corpus_ids]
        return corpus, corpus2tool

    def build_retrieval_embedder(self):
        print("Building embedder...")
        if 'simcse' in self.model_name:
            # Load in the simcse model and tokenizer if substring 'simcse' is in the model name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path).to('cuda')
        else:
            self.embedder = SentenceTransformer(self.model_path)
        
    def encode_corpus(self, sentence):
        if 'simcse' in self.model_name:
            embeddings = []
            sen_dataloader = DataLoader(sentence, batch_size=32, shuffle=False)
            for batch in tqdm(sen_dataloader, desc="Encoding sentences"):
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to('cuda')
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                    batch_embeddings = outputs.last_hidden_state[:, 0].cpu()
                    embeddings.append(batch_embeddings)
            embeddings = torch.cat(embeddings, dim=0)
            return embeddings
        else:
            return self.embedder.encode(sentence, convert_to_tensor=True)
    
    def encode_sentence(self, sentence):
        if 'simcse' in self.model_name:
            inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to('cuda')
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                embedding = outputs.last_hidden_state[:, 0].cpu()
            return embedding 
        else:
            return self.embedder.encode(sentence, convert_to_tensor=True)
    
    def build_corpus_embeddings(self):
        print("Building corpus embeddings with embedder...")
        if self.des_corpus:
            embedding_save_path = self.corpus_path.replace('.json', f'_des_corpus_{self.model_name}_embeddings.pt')
        else:
            embedding_save_path = self.corpus_path.replace('.tsv', f'_{self.model_name}_embeddings.pt')
        if os.path.exists(embedding_save_path):
            # load in previously created embeddings if the filepath exists
            print("Loading pre-computed corpus embeddings...")
            corpus_embeddings = torch.load(embedding_save_path)
            return corpus_embeddings
        print("Computing corpus embeddings...")
        corpus_embeddings = self.encode_corpus(self.corpus)

        torch.save(corpus_embeddings, embedding_save_path)
        return corpus_embeddings

    def retrieving(self, query, top_k=5, excluded_tools={}): # excluded tools is a dict, key is category, value is a list of tool names; generally not used
        # can be overriden in rapidapi.py
        print("Retrieving...")
        start = time.time()
        # run time encoding (corpus already exists)
        query_embedding = self.encode_sentence(query)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k*3, score_function=util.cos_sim)
        retrieved_tools = []
        retrieved_tool_descriptions = []
        # Additional work: take the descriptions from corpus and save them in a list
        # for multi-turn

        """
        For every hit, we enumerate it so that we have the index and the hit.
        self.corpus2tool contains descriptions as keys and category[SEP]tool_name[SEP]api_name as values.
        These values are what is returned in try statement by splitting after finding corpus2tool entry
        corresponding to the corpus_id found by rank, hit from hits 
        In order to find the correct corpus2tool entry, we use hit['corpus_id'], where corpus_id is the key.
        Now that we have the actual integer id, we go into corpus and use the hit['corpus_id'] as an index to get the description.
        This description is the key in corpus2tool, which gives us the category, tool_name, and api_name.
        The corpus embeddings are in the order of the corpus. So,
        """

        for rank, hit in enumerate(hits[0]): # in order, so the index is the rank
            try:
                category, tool_name, api_name = self.corpus2tool[self.corpus[hit['corpus_id']]].split('[SEP]')
            except Exception:
                print(f"[warn] corpus2tool lookup failed for corpus_id={hit['corpus_id']}")
                continue
            category = standardize_category(category)
            tool_name = standardize(tool_name) # standardizing
            api_name = change_name(standardize(api_name)) # standardizing
            if category in excluded_tools:
                if tool_name in excluded_tools[category]:
                    top_k += 1
                    continue
            tmp_dict = {
                "category": category,
                "tool_name": tool_name,
                "api_name": api_name,
                "corpus_id": hit['corpus_id'], # Added for multiturn
                "score": hit["score"]
            }
            retrieved_tools.append(tmp_dict)
        return retrieved_tools
    
class RemoteToolRetriever:
    """Drop-in replacement for ToolRetriever that calls a shared retriever server."""

    def __init__(self, server_url: str, corpus_name: str):
        import requests
        self.server_url = server_url.rstrip("/")
        self.corpus_name = corpus_name  # "G1", "G2", or "G3"
        self.corpus: dict = {}  # corpus_id -> description, populated lazily
        self._session = requests.Session()
        # Verify server is reachable
        resp = self._session.get(f"{self.server_url}/health")
        resp.raise_for_status()

    def retrieving(self, query, top_k=5, excluded_tools={}):
        resp = self._session.post(
            f"{self.server_url}/retrieve",
            json={"corpus": self.corpus_name, "query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        data = resp.json()
        # Update local corpus cache with descriptions from server
        for cid_str, desc in data.get("descriptions", {}).items():
            self.corpus[int(cid_str)] = desc
        return data["tools"]
