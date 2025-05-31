import argparse
import json
import os
import copy
import sys
sys.path.append('./')
os.chdir('./')
from utils.llm_proxy import LLM_Proxy
from utils.utils import format_doc,str2bool,analyze_json
from utils.es_retrieve import retrieve
from utils.utils_LLM import judge_answer_correct,clarification_generation,judge_answer_align
llm_proxy = LLM_Proxy()

model=""

judge_inquiry_type_prompt=""

answer_query_by_interaction_CoT=""
answer_query_by_interaction_vanilla=""
direct_answer_prompt=""
inquiry_generation_prompt=""
judge_uncertainty_type_prompt=""
def load_prompts(dataset):
    global judge_inquiry_type_prompt,\
        answer_query_by_interaction_CoT,answer_query_by_interaction_vanilla,direct_answer_prompt,inquiry_generation_prompt,judge_uncertainty_type_prompt
    with open('prompts/judge_uncertainty_type_by_inquiry/direct_judge_uncertainty_type_prompt.txt') as f:
        judge_inquiry_type_prompt=f.read()
    with open(f'prompts/answer_query_by_interaction/CoT/{dataset}/response_by_query_prompt.txt','r') as f:
        answer_query_by_interaction_CoT=f.read()
    with open(f'prompts/answer_query_by_interaction/vanilla/{dataset}/response_by_query_prompt.txt','r') as f:
        answer_query_by_interaction_vanilla=f.read()
    with open(f'prompts/prompt_judge_uncertainty_type/{dataset}/judge_type.txt','r') as f:
        judge_uncertainty_type_prompt=f.read()
    with open(f"prompts/direct_answer_query/{dataset}/response_by_query_prompt.txt",'r') as f:
        direct_answer_prompt=f.read()
    with open(f"prompts/inquiry_generation/{dataset}/response_by_query_prompt.txt","r") as f:
        inquiry_generation_prompt=f.read()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default='data',
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default='res',
    )
    parser.add_argument(
        "--model",
        type=str,
        default='gpt-4o',
    )
    return parser.parse_args()
def generate_answer(dataset,item):
    doc=format_doc(dataset,item)
    question=item['question']
    prompt=direct_answer_prompt.format(question,doc,question)
    flag,response=llm_proxy.llm_request(prompt)
    return response
def generate_final_answer(dataset,item,history,requires_CoT):
    question=item['question']
    doc=format_doc(dataset,item)
    if requires_CoT:
        prompt=answer_query_by_interaction_CoT.format(question,doc,question,history)
    else:
        prompt=answer_query_by_interaction_vanilla.format(question,doc,question,history)
    flag,response=llm_proxy.llm_request(prompt,model_name=model,do_sample=True)
    flag,response1=llm_proxy.llm_request(prompt,do_sample=True)
    return response,response1


def generate_inquiry(dataset,item): 
    prompt=inquiry_generation_prompt.format(item['question'],format_doc(dataset,item))
    flag,response=llm_proxy.llm_request(prompt,model_name=model)
    try:
        response=analyze_json(response)
        inquiry=response['Inquiry']
    except:
        return ""
    return inquiry

def inquiry_adaptive_answer(dataset,data,save_path=None):
    load_prompts(dataset)

    new_data=[]
    for item in data:
        history=[]
        history_inquiry=[]
        answer=None
        requires_CoT=False
        inquiry=""
        question=item['question']
        doc=format_doc(dataset,item)

        inquiry=generate_inquiry(dataset,item) # generate the inquiry
        prompt=judge_uncertainty_type_prompt.format(question,doc) # judge the source of uncertainty by prompt
        flag,choice=llm_proxy.llm_request(prompt,model_name=model,do_sample=True)

        prompt=judge_inquiry_type_prompt.format(question,inquiry) # judge the source of uncertainty by inquiry
        flag,response=llm_proxy.llm_request(prompt,do_sample=True,model_name=model)
        if "A" in response: 
            type="A"
        elif "B" in response:
            type="B"
        else:
            type="C"
        if inquiry=="":
            continue
        history_inquiry.append(inquiry)
        inquiry_score=-1
        from evaluation.generate_inquiry_evaluation import evaluate_inquiry
        inquiry_score=evaluate_inquiry(dataset,item,inquiry)
        
        if 'C' in choice: # if judgement by prompt is CoT, then we directly conduct CoT due to its high precision
            type="C"
        else:
            prompt=judge_inquiry_type_prompt.format(question,inquiry) # judge the source of uncertainty based on the inquiry
            flag,response=llm_proxy.llm_request(prompt,do_sample=True,model_name=model)
            if type in response: # the first two judgement is the same, so choose it as judgement
                pass
            else: # the first two judgement differs, so use the third judgement as the choosen one
                flag,response=llm_proxy.llm_request(prompt,do_sample=True,model_name=model)
                if "A" in response:
                    new_type="A"
                elif "B" in response:
                    new_type="B"
                elif "C" in response:
                    new_type="C"
                type=new_type

        if type=="C":
            requires_CoT=True
            type="A"
        if type not in ["A","B"]:
            continue

        from evaluation.generate_inquiry_evaluation import evaluate_inquiry
        if type=="A":

            inquiry_score=evaluate_inquiry(dataset,item,inquiry)
            if inquiry_score>=2 and item['type']=='doc': # if the inquiry points to the missing information, we provide the gold doc
                new_docs=item['gold_doc']
            else: # otherwise conduct retrieval
                new_docs=retrieve(dataset,item['question']+"\n"+inquiry,topk=2)
            # or you can also directly conduct retrieval
            # new_docs=retrieve(dataset,item['question']+"\n"+inquiry,topk=2)
            history.append({'inquiry':inquiry,'response':new_docs})
        else:

            inquiry_score=evaluate_inquiry(dataset,item,inquiry)
            if item['type']!='ambig':
                clarification='the query is clear enough'
            elif inquiry_score<2:
                clarification='the query is beyond scope'
            else:
                clarification=clarification_generation(dataset,item,inquiry)
            if "beyond scope" in clarification or "clear enough" in clarification:
                history.append({'inquiry':inquiry,'response':clarification})
            else:
                history.append({'inquiry':inquiry,'response':clarification+"\n Here are some extra documents for answering the query \n\n"+format_doc(dataset,item,gold=True)})

        answer,answer_large=generate_final_answer(dataset,item,history,requires_CoT)
        item['conversation_history']=history
        item['adaptive_answer']=answer
        item['answer_large']=answer_large
        is_answer_correct=judge_answer_correct(dataset,item['question'],item['answer'],answer)
        is_answer_large_correct=judge_answer_correct(dataset,item['question'],item['answer'],answer_large)
        item['adaptive_answer_correct']=is_answer_correct
        item['adaptive_answer_large_correct']=is_answer_large_correct
        direct_response=generate_answer(dataset,item)
        is_answer_correct=judge_answer_correct(dataset,item['question'],item['answer'],direct_response)
        item['direct_answer']=direct_response
        item['direct_answer_correct']=is_answer_correct
        item['CoT']=requires_CoT

        new_data.append(item)
    path=save_path
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{dataset}.json",'w') as f:
        json.dump(new_data,f,indent=2)
        
if __name__ == "__main__":
    args=parse_args()
    model=args.model
    datasets=['hotpotqa','ambigQA','TechQA','expertQA','toolbench']
    for dataset in datasets:
        with open(f'{args.data_path}/{dataset}.json','r') as f:
            data=json.load(f)

        inquiry_adaptive_answer(dataset,data,save_path=args.save_path)

