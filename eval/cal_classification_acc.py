import argparse
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
import json
import os
import copy
import numpy as np
import sys
sys.path.append('./')
os.chdir('./')

import argparse
from utils.utils_LLM import judge_socre_correct


def cal_classification_acc(dataset,data):
    correct,wrong=0,0
    #dic=load_train_dic(dataset)
    for item in data:
        type=item['type']

        if item['adaptive_answer']=="":
            continue
        if type=='ability':
            if item['CoT']:
                correct+=1
            elif not judge_socre_correct(dataset,item['adaptive_answer_correct']):
                wrong+=1
        elif type=='doc':
            if len(item['conversation_history'])==0: #failed conversation
                continue
            response=item['conversation_history'][-1]['response']
            if isinstance(response,str): # doc then a list, otherwise a string
                wrong+=1
            else:
                correct+=1
        else:
            if len(item['conversation_history'])==0:
                continue
            response=item['conversation_history'][-1]['response']
            if isinstance(response,str):
                correct+=1
            else:
                wrong+=1
    return correct/(correct+wrong),correct+wrong


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="toolbench",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args=parse_args()
    datasets=[args.dataset]
    for dataset in datasets:
        file=f"{args.path}/{dataset}.json"
        with open(file,'r') as f:
            data=json.load(f)
        acc,num=cal_classification_acc(dataset,data)
        f1_score=0
        print(dataset,acc,num)