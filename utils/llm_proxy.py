class LLM_Proxy:

    def __init__(self):
        self.config = {"max_length": 10000}

    def llm_request(self, prompt, do_sample=False, model_name="qwen_25_72b"):
        pass
        # if model_name=='qwen_25_72b':
        #     return self.request_qwen72b(prompt,do_sample)
