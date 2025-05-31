import json
import re
import numpy as np
import argparse


def remove_duplicate(data):
    serialized_list = [json.dumps(item, sort_keys=True) for item in data]
    print(len(serialized_list))
    unique_serialized = set(serialized_list)
    print(len(unique_serialized))
    unique_list = [json.loads(item) for item in unique_serialized]
    return unique_list


def analyze_json(text):
    starts = [
        "Here is the output in JSON format:",
        "Here is the output:",
        "Dict",
        "Here is my response:",
    ]
    for start in starts:
        if text.startswith(start):
            text = text[len(start) :]
    while text[-1] != "}" and len(text) != 0:
        text = text[0:-1]
    while text[0] != "{" and len(text) != 0:
        text = text[1:]
    if text.startswith("```json"):
        text = text[7:-3]
    try:
        text = json.loads(text)
        return text
    except:
        return None


def split_train_test(data, p=0.7):
    length = len(data)
    idx = np.arange(length)
    idx = np.random.permutation(idx)
    data = [data[i] for i in idx]
    train_num = (int)(p * length)
    train_data = data[0:train_num]
    test_data = data[train_num:]
    return train_data, test_data


def format_doc(dataset, item, gold=False, document=None):
    if document != None:
        doc = document
    elif gold:
        doc = item["gold_doc"]
    else:
        doc = item["doc"]

    for i in range(len(doc)):
        idx = doc[i].find("'paragraph_text':")
        if idx == -1:
            continue
        idx += 19
        doc[i] = doc[i][idx:-2]
    if not gold and document == None:
        len_gold = len(item["gold_doc"])
        if dataset == "hotpotqa":
            start = -1 * (len_gold + 2)
        else:
            start = -1 * (len_gold + 2)
        if item["type"] == "ambig" and dataset != "toolbench":
            doc = doc[start : -1 * len_gold]
        else:
            doc = doc[-len_gold:]
    if dataset == "toolbench":
        doc = "[API]: " + " [API]: ".join(doc)
    else:
        doc = "[Document]: " + " [Document]: ".join(doc)
    return doc


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
