import argparse
import json
import os
import copy
import sys

sys.path.append("./")
os.chdir("./")
from utils.llm_proxy import LLM_Proxy
from utils.utils import format_doc, str2bool, analyze_json
from utils.es_retrieve import retrieve
from utils.utils_LLM import (
    judge_answer_correct,
    clarification_generation,
    judge_answer_align,
)

llm_proxy = LLM_Proxy()
model = ""
generate_inquiry_answer_prompt = ""
judge_inquiry_type_prompt = ""
judge_inquiry_quality_prompt = ""
answer_query_by_interaction_CoT = ""
answer_query_by_interaction_vanilla = ""
inquiry_generation_prompt = ""
judge_uncertainty_type_prompt = ""
direct_answer_prompt = ""


def load_prompts(dataset):
    global direct_answer_prompt, generate_inquiry_answer_prompt, judge_inquiry_type_prompt, judge_inquiry_quality_prompt, answer_query_by_interaction_CoT, answer_query_by_interaction_vanilla, inquiry_generation_prompt, judge_uncertainty_type_prompt, generate_inquiry_answer_prompt_CoT
    with open(f"prompts/generate_inquiry_answer/generate_inquiry_answer_prompt.txt") as f:
        generate_inquiry_answer_prompt = f.read()
    with open("prompts/judge_uncertainty_type_by_inquiry/direct_judge_uncertainty_type_prompt.txt") as f:
        judge_inquiry_type_prompt = f.read()
    with open("prompts/judge_uncertainty_type_by_inquiry/judge_inquiry_quality_prompt.txt") as f:
        judge_inquiry_quality_prompt = f.read()
    with open(f"prompts/answer_query_by_interaction/CoT/{dataset}/response_by_query_prompt.txt","r",) as f:
        answer_query_by_interaction_CoT = f.read()
    with open(f"prompts/prompt_judge_uncertainty_type/{dataset}/judge_type.txt", "r") as f:
        judge_uncertainty_type_prompt = f.read()
    with open(f"prompts/direct_answer_query/{dataset}/response_by_query_prompt.txt", "r") as f:
        direct_answer_prompt = f.read()
    with open(f"prompts/answer_query_by_interaction/vanilla/{dataset}/response_by_query_prompt.txt","r",) as f:
        answer_query_by_interaction_vanilla = f.read()
    with open(f"prompts/inquiry_generation/{dataset}/response_by_query_prompt.txt", "r") as f:
        inquiry_generation_prompt = f.read()

repetition_num = 1
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="res",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
    )
    parser.add_argument(
        "--repetition_num",
        type=int,
        default=1,
    )
    return parser.parse_args()


def generate_answer(dataset, item): # use Qwen72b to directly generate answer without interaction
    doc = format_doc(dataset, item)
    question = item["question"]
    prompt = direct_answer_prompt.format(question, doc, question)
    flag, response = llm_proxy.llm_request(prompt)
    return response


def generate_final_answer(dataset, item, history, requires_CoT): # use evaluated model and Qwen72b to generate the answer with interaction
    question = item["question"]
    doc = format_doc(dataset, item)
    if requires_CoT: # use CoT
        prompt = answer_query_by_interaction_CoT.format(
            question, doc, question, history
        )
    else: # vanilla generation
        prompt = answer_query_by_interaction_vanilla.format(
            question, doc, question, history
        )
    flag, response = llm_proxy.llm_request(prompt, model_name=model, do_sample=True)
    flag, response1 = llm_proxy.llm_request(prompt, do_sample=True)
    return response, response1


with open("utils/prompts/judge_answer_fit_prompt.txt", "r") as f:
    judge_answer_fit_prompt = f.read()


def judge_answer_fit(original_question, generated_answer, model): # judge does the answer to the inquiry fits the original query
    prompt = judge_answer_fit_prompt.format(original_question, generated_answer)
    flag, response = llm_proxy.llm_request(prompt, model_name=model)
    if "yes" in response.lower():
        return True
    return False


def judge_by_answer(dataset, item, inquiry, C_only=True):
    global repetition_num
    question = item["question"]
    previous_answer = []
    prompt = generate_inquiry_answer_prompt.format(question, format_doc(dataset, item), inquiry, previous_answer)
    flag, response = llm_proxy.llm_request(prompt, do_sample=True, model_name=model)  # generate the first answer
    previous_answer = [response] * repetition_num

    prompt = judge_inquiry_quality_prompt.format(question, inquiry)
    flag, response = llm_proxy.llm_request(
        prompt, do_sample=True, model_name=model
    )  # judge the quality of the inquiry
    low_quality = False
    if "useless" in response.lower():
        low_quality = True
    if low_quality:
        fit = judge_answer_fit(
            question, previous_answer[0], model
        )  # low quality and the answer fits the original query, then CoT
        if fit:
            return "C"
    if C_only:
        return None
    prompt = generate_inquiry_answer_prompt.format(question, format_doc(dataset, item), inquiry, previous_answer)  # generate second answer
    flag, response = llm_proxy.llm_request(prompt, do_sample=True, model_name=model)
    correct = judge_answer_align(inquiry, previous_answer[0], response, model)  # judge does the two answer aligns
    if correct:
        return "A"
    else:
        return "B"


def generate_inquiry(dataset, item):  # generate the inquiry
    prompt = inquiry_generation_prompt.format(item["question"], format_doc(dataset, item))
    flag, response = llm_proxy.llm_request(prompt, model_name=model)
    try:
        response = analyze_json(response)
        inquiry = response["Inquiry"]
    except:
        return ""
    return inquiry


def inquiry_adaptive_answer(dataset, data, save_path=None):
    load_prompts(dataset)

    new_data = []
    for item in data:
        history = []
        history_inquiry = []
        answer = None
        requires_CoT = False
        inquiry = ""
        question = item["question"]
        doc = format_doc(dataset, item)

        inquiry = generate_inquiry(dataset, item)

        prompt = judge_uncertainty_type_prompt.format(
            question, doc
        )  # judge the source of uncertainty by prompt
        flag, choice = llm_proxy.llm_request(prompt, model_name=model, do_sample=True)

        prompt = judge_inquiry_type_prompt.format(
            question, inquiry
        )  # judge the source of uncertainty by inquiry
        flag, response = llm_proxy.llm_request(prompt, do_sample=True, model_name=model)
        if "A" in response:
            type = "A"
        elif "B" in response:
            type = "B"
        else:
            type = "C"

        if inquiry == "":
            continue
        history_inquiry.append(inquiry)

        inquiry_score = -1

        if ("C" in choice):  # if judgement by prompt is CoT, then we directly conduct CoT due to its high precision,
            type = "C"
        else:
            prompt = judge_inquiry_type_prompt.format(question, inquiry)  # judge based on the inquiry
            flag, response = llm_proxy.llm_request(prompt, do_sample=True, model_name=model)
            if (type in response):  # if the first two judgement based on inquiry is the same, then verify using judge by answer to identify is the inquiry meaningful
                new_type = judge_by_answer(dataset, item, inquiry, C_only=True)
                if new_type == "C":
                    type = new_type
            else:

                new_type = judge_by_answer(dataset, item, inquiry, C_only=True)
                if new_type == "C":
                    pass
                else:
                    flag, response = llm_proxy.llm_request(
                        prompt, do_sample=True, model_name=model
                    )
                    if "A" in response:
                        new_type = "A"
                    elif "B" in response:
                        new_type = "B"
                    else:
                        new_type = "C"
                type = new_type

        if type == "C":
            requires_CoT = True
            type = "A"
        if type not in ["A", "B"]:
            continue
        from evaluation.generate_inquiry_evaluation import evaluate_inquiry

        if type == "A":

            inquiry_score = evaluate_inquiry(dataset, item, inquiry)
            if inquiry_score>=2 and item['type']=='doc':  # if the inquiry points to the missing information, we provide the gold doc
                new_docs = item["gold_doc"]
            else:  # otherwise conduct retrieval
                new_docs = retrieve(dataset, item["question"] + "\n" + inquiry, topk=2)
            # or you can also directly conduct retrieval
            # new_docs=retrieve(dataset,item['question']+"\n"+inquiry,topk=2)
            history.append({"inquiry": inquiry, "response": new_docs})
        else:

            inquiry_score = evaluate_inquiry(dataset, item, inquiry)
            if item["type"] != "ambig":
                clarification = "the query is clear enough"
            elif inquiry_score < 2:
                clarification = "the query is beyond scope"
            else:
                clarification = clarification_generation(dataset, item, inquiry)
            if "beyond scope" in clarification or "clear enough" in clarification:
                history.append({"inquiry": inquiry, "response": clarification})
            else:
                history.append(
                    {
                        "inquiry": inquiry,
                        "response": clarification
                        + "\n Here are some extra documents for answering the query \n\n"
                        + format_doc(dataset, item, gold=True),
                    }
                )

        answer, answer_large = generate_final_answer(
            dataset, item, history, requires_CoT
        )
        item["conversation_history"] = history
        item["adaptive_answer"] = answer
        item["answer_large"] = answer_large
        is_answer_correct = judge_answer_correct(
            dataset, item["question"], item["answer"], answer
        )
        is_answer_large_correct = judge_answer_correct(
            dataset, item["question"], item["answer"], answer_large
        )
        item["adaptive_answer_correct"] = is_answer_correct
        item["adaptive_answer_large_correct"] = is_answer_large_correct
        direct_response = generate_answer(dataset, item)
        is_answer_correct = judge_answer_correct(
            dataset, item["question"], item["answer"], direct_response
        )
        item["direct_answer"] = direct_response
        item["direct_answer_correct"] = is_answer_correct

        item['CoT']=requires_CoT
        new_data.append(item)
    path = save_path
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{dataset}.json", "w") as f:
        json.dump(new_data, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    model = args.model
    repetition_num = args.repetition_num
    datasets = ['hotpotqa','ambigQA','TechQA','expertQA','toolbench']
    for dataset in datasets:

        with open(f"{args.data_path}/{dataset}.json", "r") as f:
            data = json.load(f)

        inquiry_adaptive_answer(dataset, data, save_path=args.save_path)
