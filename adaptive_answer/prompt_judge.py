import argparse
import json
import os
import copy
import sys

sys.path.append("./")
os.chdir("./")
from utils.utils import str2bool
from utils.llm_proxy import LLM_Proxy
from utils.utils import format_doc, analyze_json
from utils.es_retrieve import retrieve
from utils.utils_LLM import judge_answer_correct, clarification_generation
from evaluation.generate_inquiry_evaluation import evaluate_inquiry

llm_proxy = LLM_Proxy()

answer_query_by_interaction_CoT = ""
answer_query_by_interaction_vanilla = ""
direct_answer_prompt = ""
judge_uncertainty_type_prompt = ""
generate_inquiry_d_prompt = ""
generate_inquiry_c_prompt = ""


def load_prompts(dataset):  # loading prompts
    global answer_query_by_interaction_CoT, answer_query_by_interaction_vanilla, direct_answer_prompt, judge_uncertainty_type_prompt, generate_inquiry_c_prompt, generate_inquiry_d_prompt

    with open(f"prompts/answer_query_by_interaction/CoT/{dataset}/response_by_query_prompt.txt","r",) as f:
        answer_query_by_interaction_CoT = f.read()
    with open(f"prompts/answer_query_by_interaction/vanilla/{dataset}/response_by_query_prompt.txt","r",) as f:
        answer_query_by_interaction_vanilla = f.read()
    with open(f"prompts/direct_answer_query/{dataset}/response_by_query_prompt.txt", "r") as f:
        direct_answer_prompt = f.read()

    with open(f"prompts/prompt_judge_uncertainty_type/generate_inquiry_d.txt", "r") as f:
        generate_inquiry_d_prompt = f.read()
    with open(f"prompts/prompt_judge_uncertainty_type/generate_inquiry_c.txt", "r") as f:
        generate_inquiry_c_prompt = f.read()

    with open(f"prompts/prompt_judge_uncertainty_type/{dataset}/judge_type.txt", "r") as f:
        judge_uncertainty_type_prompt = f.read()



model = ""
def parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def generate_answer(dataset, item): # use Qwen72b to directly generate answer without interaction
    doc = format_doc(dataset, item)
    question = item["question"]
    prompt = direct_answer_prompt.format(question, doc, question)
    flag, response = llm_proxy.llm_request(prompt)
    return response


def generate_choice(
    dataset, item, history
):  # judge the source of uncertainty for 3 times
    question = item["question"]
    doc = format_doc(dataset, item)

    prompt = judge_uncertainty_type_prompt.format(question, doc)
    flag, choice0 = llm_proxy.llm_request(prompt, model_name=model, do_sample=True)
    try:
        choice = choice0[0]
    except:
        choice=choice0

    if choice != "C":
        flag, choice1 = llm_proxy.llm_request(prompt, model_name=model, do_sample=True)
        flag, choice2 = llm_proxy.llm_request(prompt, model_name=model, do_sample=True)
        try:
            choice1, choice2 = choice1[0], choice2[0]
        except:
            pass
        if choice1 == choice2:
            choice = choice1
        else:
            flag, choice = llm_proxy.llm_request(prompt, model_name=model, do_sample=True)

    if choice == "C": # generate corresponding inquiry based on the source of uncertainty
        generate_inquiry_prompt = generate_inquiry_d_prompt.format(question, doc)
    elif choice == "A":
        generate_inquiry_prompt = generate_inquiry_d_prompt.format(question, doc)
    elif choice == "B":
        generate_inquiry_prompt = generate_inquiry_c_prompt.format(question, doc)
    else:
        return choice, ""
    flag, inquiry = llm_proxy.llm_request(generate_inquiry_prompt, model_name=model)
    return choice, inquiry


def generate_final_answer(dataset, item, history, requires_CoT): # use evaluated model and Qwen72b to generate the answer with interaction
    question = item["question"]
    doc = format_doc(dataset, item)
    if requires_CoT: # use CoT
        prompt = answer_query_by_interaction_CoT.format(question, doc, question, history)
    else: # vanilla generation
        prompt = answer_query_by_interaction_vanilla.format(question, doc, question, history)
    flag, response = llm_proxy.llm_request(prompt, model_name=model, do_sample=True)
    flag, response1 = llm_proxy.llm_request(prompt,model_name='gpt-4o-0513', do_sample=True)
    return response, response1


def adaptive_generate_answer(dataset, data, save_path):
    load_prompts(dataset)
    new_data = []
    for item in data:
        inquiry_score = -1
        history = []
        answer = None
        requires_CoT = False
        inquiry = ""
        choice, inquiry = generate_choice(dataset, item, history)  # judge the source of uncertainty and generate inquiry
        inquiry_score = evaluate_inquiry(dataset, item, inquiry)  # evaluate the quality of inquiry

        if choice == "C":  # if requires CoT, we also conduct retrieval
            requires_CoT = True
            choice = "A"

        if choice == "A":

            if inquiry_score>=2 and item['type']=='doc':  # if the inquiry points to the missing information, we provide the gold doc
                new_docs = item["gold_doc"]
            else:  # otherwise conduct retrieval
                new_docs = retrieve(dataset, item["question"] + "\n" + inquiry, topk=2)
            # or you can also directly conduct retrieval
            # new_docs=retrieve(dataset,item['question']+"\n"+inquiry,topk=2)
            history.append({"inquiry": inquiry, "response": new_docs})

        elif choice == "B":
            if item["type"] != "ambig":
                clarification = "the query is clear enough"
            elif inquiry_score < 2:
                clarification = "the query is beyond scope" # if the inquiry is irrelevant, directly response with  beyond scope
            else:
                clarification = clarification_generation(dataset, item, inquiry) # generation clarification
            if "beyond scope" in clarification or "clear enough" in clarification:
                history.append({"inquiry": inquiry, "response": clarification})
            else:
                history.append(
                    {
                        "inquiry": inquiry,
                        "response": clarification
                        + "\n Here are some extra documents for answering the query \n\n"
                        + format_doc(dataset, item, gold=True), # get the gold documents to answer the ambiguous query
                    }
                )
        else:
            pass
        answer, answer_large = generate_final_answer(
            dataset, item, history, requires_CoT
        )

        # record the information into the dict
        item["conversation_history"] = history
        item["adaptive_answer"] = answer
        item["answer_large"] = answer_large
        is_answer_correct = judge_answer_correct(dataset, item["question"], item["answer"], answer)
        is_answer_large_correct = judge_answer_correct(dataset, item["question"], item["answer"], answer_large)
        item["adaptive_answer_correct"] = is_answer_correct
        item["adaptive_answer_large_correct"] = is_answer_large_correct
        direct_response = generate_answer(dataset, item) # judge can the question be directly answered
        is_answer_correct = judge_answer_correct(dataset, item["question"], item["answer"], direct_response)
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
    datasets = ['hotpotqa','ambigQA','TechQA','expertQA','toolbench']
    for dataset in datasets:
        with open(f"{args.data_path}/{dataset}.json", "r") as f:
            data = json.load(f)
        adaptive_generate_answer(dataset, data, save_path=args.save_path)
