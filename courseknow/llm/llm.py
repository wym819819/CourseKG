import os
import requests
from modelscope import AutoTokenizer, AutoModelForCausalLM


class LLM:

    def __init__(self) -> None:
        """ 多种大模型封装类
        """

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入
        
        Raises:
            NotImplementedError: 子类需要实现该方法

        Returns:
            str: 模型输出
        """
        raise NotImplementedError


class QwenAPI(LLM):

    def __init__(self,
                 api_type: str = 'qwen-max',
                 api_key: str = None) -> None:
        """ Qwen 系列模型 API 服务

        Args:
            api_type (str, optional): 模型类型. Defaults to 'qwen-max'.
            api_key (str, optional): API_KEY, 不输入则尝试从环境变量中获取. Defaults to None.
        """
        self.api_type = api_type
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_key = api_key

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        body = {
            'model': self.api_type,
            "input": {
                "messages": [{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role": "user",
                    "content": message
                }]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        response = requests.post(url, headers=headers, json=body)
        return response.json()['output']['choices'][0]['message']['content']


class Qwen2(LLM):

    def __init__(self, path: str) -> None:
        """ Qwen2 系列模型

        Args:
            path (str): 模型名称或路径
        """
        self.path = path
        self.device = 'cuda'
        self.model = None
        self.tokenizer = None

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path, torch_dtype="auto", device_map="auto")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role": "user",
            "content": message
        }]
        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        model_inputs = self.tokenizer([text],
                                      return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(model_inputs.input_ids,
                                            max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids,
                                               skip_special_tokens=True)[0]
        return response
