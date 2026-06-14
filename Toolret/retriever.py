import json
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os


def process_des_retrieval_document(documents_path):
    ir_corpus = []
    corpus2tool = {}
    with open(documents_path, 'r') as f:
        for line in f:
            tool_json = json.loads(line)
            ir_corpus.append(tool_json["description"])
            corpus2tool[tool_json["description"]] = tool_json['id']
    return ir_corpus, corpus2tool

class ToolRetriever:
    def __init__(self, corpus_path = "", model_path=""):
        self.corpus_path = corpus_path
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.corpus, self.corpus2tool = self.build_retrieval_corpus()
        self.build_retrieval_embedder()
        self.corpus_embeddings = self.build_corpus_embeddings()

    def build_retrieval_corpus(self):
        print("Building corpus...")
        corpus, corpus2tool = process_des_retrieval_document(self.corpus_path)
        return corpus, corpus2tool

    def build_retrieval_embedder(self):
        print("Building embedder...")
        if 'simcse' in self.model_name:
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
        embedding_save_path = self.corpus_path.replace('.json', f'_des_corpus_{self.model_name}_embeddings.pt')
        if os.path.exists(embedding_save_path):
            print(f"Loading existing corpus embeddings from {embedding_save_path}")
            corpus_embeddings = torch.load(embedding_save_path)
            return corpus_embeddings
        print("Encoding corpus...")
        corpus_embeddings = self.encode_corpus(self.corpus)

        torch.save(corpus_embeddings, embedding_save_path)
        print(f"Saved corpus embeddings to {embedding_save_path}")
        return corpus_embeddings
    
    def retrieving(self, query, top_k=5):
        print("Retrieving tools...")
        query_embedding = self.encode_sentence(query)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k, score_function=util.cos_sim)
        retrieved_tools_ids = []
        retrieved_tools_des = []
        retrieved_tools_score = []
        for hit in hits[0]:
            try:
                tool_description = self.corpus[hit['corpus_id']]
                tool_id = self.corpus2tool[tool_description]
                retrieved_tools_ids.append(tool_id)
                retrieved_tools_des.append(tool_description)
                retrieved_tools_score.append(hit['score'])
            except Exception as e:
                print(f"Error retrieving tool for hit {hit}: {e}")
        return retrieved_tools_ids, retrieved_tools_des, retrieved_tools_score


