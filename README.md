

## Data
The benchmark dataset is shown in data/
* question: the question
* gold doc: gold documents which directly helps answer the question
* doc: actual docs will be provided to the model
* answer: ground truth
* original question: if the query is ambiguous, then it stands for the original query, otherwise null
* type: "doc" for lack of documents, "ambig" for ambiguous query, "ability" for lack of capacitys

## Run
get corpus for the datasets
* HotpotQA from https://github.com/starsuzi/Adaptive-RAG
* AmbigQA from https://github.com/shmsw25/AmbigQA
* TechQA from https://github.com/ibm/techqa
* ExpertQA from https://github.com/chaitanyamalaviya/expertqa
* ToolBench from https://github.com/OpenBMB/ToolBench

### prepare retriever server
```
$ wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
$ wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
$ shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
$ tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
$ cd elasticsearch-7.10.2/
$ ./bin/elasticsearch # start the server
# pkill -f elasticsearch # to stop the server
```
start the elastic search server by

`uvicorn serve:app --port 8000 --app-dir retriever_server`

put the corpus, index and retriever model of toolbench in `toolbench_retriever`

start toolbench retriever server in 
`python toolbench_retriever/toolbench_retriever_server.py --port {}`

set the toolbench retriever port in `utils/es_retrieve`

Set the llm calling API in `utils/llm_proxy`

### evaluation

run the following command to judge the source of uncertainty by prompt, inquiry, and the answer of inquiry

```
python adaptive_answer/prompt_judge.py --model {}
python adaptive_answer/inquiry_judge.py --model {}
python adaptive_answer/answer_judge.py --model {}
```
