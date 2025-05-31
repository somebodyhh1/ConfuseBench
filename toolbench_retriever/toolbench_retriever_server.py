
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"
import torch
from peft import get_peft_model,LoraConfig
from fastapi import FastAPI
from pydantic import BaseModel
import json
from elasticsearch import Elasticsearch
import json
import re
import yaml
from sentence_transformers import SentenceTransformer,models
import faiss
import pandas as pd
import pickle
from faiss import normalize_L2
import os
import numpy as np
with open('toolbench_retriever/corpus_list.pkl','rb') as f:
    toolbench_corpus=pickle.load(f)


app = FastAPI()
class PredictInput(BaseModel):
    query: str
    topk: int

word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
toolbench_retrieve_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
toolbench_retrieve_model.load('toolbench_retriever/model')
toolbench_vector=faiss.read_index('toolbench_retriever/vectorstore/toolbench.index')

@app.post('/retrieve')
def retrieve(input:PredictInput):
    query=input.query
    topk=input.topk
    query_embed=toolbench_retrieve_model.encode([query])
    normalize_L2(query_embed)
    _, match_id=toolbench_vector.search(query_embed,topk)
    results=[]
    for i in match_id[0]:
        tool=toolbench_corpus[i]
        #print("keys==",tool.keys())
        dic={"name":tool['api_name'],'description':f"This is the subfunction for tool '{tool['tool_name']}', you can use this tool.The description of this function is: {tool['api_description']}",'parameters':tool['required_parameters']}
        results.append(dic)
    return [json.dumps(results[i]) for i in range(len(results))]


import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=None,
    )
    return parser.parse_args()
if __name__ == '__main__':
    args=parse_args()
    port=args.port
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port)


