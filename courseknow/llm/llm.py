import os
import requests
from modelscope import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod
import vllm
from vllm import SamplingParams


class LLM(ABC):

    def __init__(self) -> None:
        """ 多种大模型封装类
        """
        pass

    @abstractmethod
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
            api_key (str, optional): API_KEY, 不输入则尝试从环境变量 DASHSCOPE_API_KEY 中获取. Defaults to None.
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
        self.llm = vllm.LLM(model=path, tensor_parallel_size=2)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        sampling_params = SamplingParams(temperature=0.7,
                                         top_p=0.8,
                                         repetition_penalty=1.05,
                                         max_tokens=2048)
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

        outputs = self.llm.generate([text], sampling_params)
        return outputs[0].outputs[0].text
