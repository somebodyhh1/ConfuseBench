import json
import os
import copy
import sys

sys.path.append("./")
os.chdir("./")
from utils.llm_proxy import LLM_Proxy

llm_proxy = LLM_Proxy()
with open("utils/prompts/judge_answer_correct_prompt.txt", "r") as f:
    judge_answer_correct_prompt = f.read()
with open("utils/prompts/judge_answer_align_prompt.txt", "r") as f:
    judge_answer_align_prompt = f.read()
with open("utils/prompts/evaluate_answer_score_prompt.txt", "r") as f:
    evaluate_answer_score_prompt = f.read()
with open("utils/prompts/evaluate_answer_score_toolbench_prompt.txt", "r") as f:
    evaluate_answer_score_toolbench_prompt = f.read()
with open("utils/prompts/clarification_generation_prompt.txt", "r") as f:
    clarification_generation_prompt = f.read()
with open("utils/prompts/clarification_rewrite_prompt.txt", "r") as f:
    clarification_rewrite_prompt = f.read()


def judge_socre_correct(dataset, score):
    if dataset in ["hotpotqa", "ambigQA"]:
        if score == 1:
            return True
        return False
    else:
        if score >= 4:
            return True
        return False


def judge_answer_correct(
    dataset, question, answer, response, score=None, model="gpt-4o-0513"
):

    if dataset in ["hotpotqa", "ambigQA"] or score == False:
        judge_prompt = judge_answer_correct_prompt.format(question, answer, response)
        flag, response = llm_proxy.llm_request(
            judge_prompt, judge=True, model_name=model
        )
        if "true" in response.lower():
            return 1
        if "false" in response.lower():
            return 0
        return -1
    elif dataset in ["toolbench"]:
        judge_prompt = evaluate_answer_score_toolbench_prompt.format(
            question, answer, response
        )
        flag, response = llm_proxy.llm_request(
            judge_prompt, judge=True, model_name=model
        )
        try:
            score = (int)(response)
            return score
        except:
            return -1
    else:
        judge_prompt = evaluate_answer_score_prompt.format(question, answer, response)
        flag, response = llm_proxy.llm_request(judge_prompt, model_name=model)
        try:
            score = (int)(response)
            return score
        except:
            return -1


def judge_answer_align(inquiry, answer1, answer2, model):
    judge_prompt = judge_answer_align_prompt.format(inquiry, answer1, answer2)
    flag, response = llm_proxy.llm_request(judge_prompt, model_name=model)
    if "true" in response.lower():
        return 1
    if "false" in response.lower():
        return 0
    return -1


def clarification_generation(dataset, item, inquiry):
    question = item["question"]
    original_question = item["original_query"]
    if original_question is None:
        original_question = question
    prompt = copy.deepcopy(clarification_generation_prompt)
    prompt = prompt.format(original_question, question, inquiry)
    flag, response = llm_proxy.llm_request(prompt, model_name="gpt-4o-mini")
    return response


def clarification_rewrite(dataset, item, clarification, model):
    prompt = clarification_rewrite_prompt.format(item["question"], clarification)
    flag, response = llm_proxy.llm_request(prompt, model_name=model)
    return response
