import os


import json
from elasticsearch import Elasticsearch
import json
import re
import yaml
from sentence_transformers import SentenceTransformer, models
import faiss
import pandas as pd
import pickle
from faiss import normalize_L2
import os
import numpy as np
import requests

toolbench_retrieve_model = None
toolbench_vector = None
TOOLBENCH_PORT=9096

core_title_matcher = re.compile("([^()]+[^\s()])(?:\s*\(.+\))?")
core_title_filter = lambda x: (
    core_title_matcher.match(x).group(1) if core_title_matcher.match(x) else x
)


class ElasticSearch:
    def __init__(self, index_name):
        self.index_name = index_name
        self.client = Elasticsearch("http://localhost:9200")

    def _extract_one(self, item, lazy=False):
        res = {
            k: item["_source"][k]
            for k in ["id", "url", "title", "text", "title_unescape"]
        }
        res["_score"] = item["_score"]
        return res

    def rerank_with_query(self, query, results):
        def score_boost(item, query):
            score = item["_score"]
            core_title = core_title_filter(item["title_unescape"])
            if query.startswith("The ") or query.startswith("the "):
                query1 = query[4:]
            else:
                query1 = query
            if query == item["title_unescape"] or query1 == item["title_unescape"]:
                score *= 1.5
            elif (
                query.lower() == item["title_unescape"].lower()
                or query1.lower() == item["title_unescape"].lower()
            ):
                score *= 1.2
            elif item["title"].lower() in query:
                score *= 1.1
            elif query == core_title or query1 == core_title:
                score *= 1.2
            elif (
                query.lower() == core_title.lower()
                or query1.lower() == core_title.lower()
            ):
                score *= 1.1
            elif core_title.lower() in query.lower():
                score *= 1.05

            item["_score"] = score
            return item

        return list(
            sorted(
                [score_boost(item, query) for item in results],
                key=lambda item: -item["_score"],
            )
        )

    def retrieve_medqa(self, query, topk):
        constructed_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "sentence",
                ],
            }
        }
        res = self.client.search(
            index=self.index_name,
            body={"query": constructed_query, "timeout": "100s"},
            size=topk,
            request_timeout=100,
        )

        res = [x["_source"] for x in res["hits"]["hits"]]
        res = [{"paragraph_text": _["sentence"]} for _ in res]
        return res

    def retrieve_expertqa(self, query, topk):
        constructed_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "text",
                ],
            }
        }
        res = self.client.search(
            index=self.index_name,
            body={"query": constructed_query, "timeout": "100s"},
            size=topk,
            request_timeout=100,
        )

        res = [x["_source"] for x in res["hits"]["hits"]]
        res = [{"paragraph_text": _["text"]} for _ in res]
        return res

    def retrieve_ambigqa(self, query, topk):
        constructed_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "text",
                ],
            }
        }
        res = self.client.search(
            index=self.index_name,
            body={"query": constructed_query, "timeout": "100s"},
            size=topk,
            request_timeout=100,
        )

        res = [x["_source"] for x in res["hits"]["hits"]]
        res = [{"paragraph_text": _["text"]} for _ in res]
        return res

    def retrieve_techqa(self, query, topk):
        constructed_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^1.25",
                    "text",
                ],
            }
        }
        res = self.client.search(
            index=self.index_name,
            body={"query": constructed_query, "timeout": "100s"},
            size=topk,
            request_timeout=100,
        )
        res = [x["_source"] for x in res["hits"]["hits"]]

        res = [{"title": _["title"], "paragraph_text": _["text"]} for _ in res]
        return res

    def retrieve_hotpotqa(self, query, topn=10, lazy=False, rerank_topn=50):

        constructed_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "title^1.25",
                    "title_unescape^1.25",
                    "text",
                    "title_bigram^1.25",
                    "title_unescape_bigram^1.25",
                    "text_bigram",
                ],
            }
        }
        res = self.client.search(
            index=self.index_name,
            body={"query": constructed_query, "timeout": "100s"},
            size=max(topn, rerank_topn),
            request_timeout=100,
        )

        res = [self._extract_one(x, lazy=lazy) for x in res["hits"]["hits"]]
        res = self.rerank_with_query(query, res)[:topn]
        res = [{"title": _["title"], "paragraph_text": _["text"]} for _ in res]
        return res

    def search(self, index_name, question, k=10):
        try:
            if index_name == "hotpotqa1":
                res = self.retrieve_hotpotqa(query=question, topn=k)
                return json.dumps(res, ensure_ascii=False)
            if index_name == "medqa":
                res = self.retrieve_medqa(question, k)
                return json.dumps(res, ensure_ascii=False)
            if index_name == "techqa":
                res = self.retrieve_techqa(question, k)
                return json.dumps(res, ensure_ascii=False)
            if index_name == "expertqa":
                res = self.retrieve_expertqa(question, k)
                return json.dumps(res, ensure_ascii=False)
            if index_name == "ambigqa":
                res = self.retrieve_ambigqa(question, k)
                return json.dumps(res, ensure_ascii=False)
        except Exception as err:
            print(Exception, err)
            raise


def retrieve_toolbench(query, topk):

    url = f"http://0.0.0.0:{TOOLBENCH_PORT}/retrieve"
    input = {"query": query, "topk": topk}
    try:
        response = requests.post(url, json=input)  # 将数据以 JSON 格式发送
        # 检查响应状态
        if response.status_code == 200:
            # 请求成功，处理返回结果
            result = response.json()  # 将响应内容解析为 JSON

            return result
        else:
            print("response error==", response)
            # 请求失败，处理错误
            return []

    except requests.exceptions.RequestException as e:
        # 捕获请求过程中可能发生的异常
        print(f"请求过程中发生错误: {e}")
    return []


import logging


def retrieve(index_name, query, topk):
    try:
        index_name = index_name.lower()
        if index_name == "hotpotqa":
            index_name = "hotpotqa1"
        if index_name == "test":
            return [""]
        elif index_name != "toolbench":
            ES = ElasticSearch(index_name)
            result = ES.search(index_name, query, topk)
            result = json.loads(result)
            result = [str(result[i]["paragraph_text"]) for i in range(len(result))]
            return result
        else:
            return retrieve_toolbench(query, topk)
    except BaseException as e:
        print("meet retrieval exception", e)
        logging.exception(e)
        return []


if __name__ == "__main__":
    print(
        retrieve(
            "hotpotqa1",
            "Who is Colin Kaepernick and what is his preferred nickname?",
            2,
        )
    )
