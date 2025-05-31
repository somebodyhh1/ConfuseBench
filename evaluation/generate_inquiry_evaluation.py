import argparse
import json
import os
import copy
import sys
sys.path.append('./')
os.chdir('./')
from utils.llm_proxy import LLM_Proxy
from utils.utils import format_doc,analyze_json
llm_proxy=LLM_Proxy()
import argparse

with open("prompts/evaluation/judge_inquiry_relevance_prompt.txt",'r') as f:
    judge_inquiry_relevance_prompt=f.read()
with open("prompts/evaluation/gold_inquiry_generation_prompt.txt",'r') as f:
    gold_inquiry_generation_prompt=f.read()


def generate_gold_inquiry(dataset,item,model='gpt-4o-mini'):
    question=item['question']
    original_question=question if item['original_query'] is None else item['original_query']
    doc=format_doc(dataset,item)
    gold_doc=format_doc(dataset,item,gold=True)
    type=item['type']
    if type=="ambig":
        prompt=gold_inquiry_generation_prompt.format(original_question,"",question,"")
    else:
        prompt=gold_inquiry_generation_prompt.format("",gold_doc,"",doc)
    flag,response=llm_proxy.llm_request(prompt,model_name=model)
    return response

def evaluate_inquiry(dataset,item,inquiry,model='gpt-4o-mini'):
    gold_inquiry=generate_gold_inquiry(dataset,item,model='qwen72b')
    question=item['question']
    original_question=question if item['original_query'] is None else item['original_query']
    gold_doc=format_doc(dataset,item,gold=True)
    doc=format_doc(dataset,item)
    if inquiry=="":
        return -1
    prompt=judge_inquiry_relevance_prompt.format(original_question,gold_doc,question,doc,gold_inquiry,inquiry)
    for _ in range(5):
        flag,response=llm_proxy.llm_request(prompt,model_name=model)
        try:
            res=analyze_json(response)
            score=res['quality of inquiry']
            score=(int)(score)
            return score
        except:
            continue
    return -1
