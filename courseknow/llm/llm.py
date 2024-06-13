import os
from enum import Enum
import requests
from modelscope import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                        GenerationConfig, BitsAndBytesConfig)
import torch


class LLMType(Enum):
    Qwen_Turbo = 'qwen-turbo'
    Qwen_Max = 'qwen-max'
    LocalQwen2_7B = 'Qwen/Qwen2-7B-Instruct'
    LocalOneKE = 'zjunlp/OneKE'


class LLM:

    def __init__(self, llm_type: LLMType, api_key: str = None) -> None:
        """ 调用 LLM 的 API 服务

        Args:
            llm_type (LLMType): 模型类型
            api_key (str, optional): API_KEY, 不输入则尝试从环境变量中获取. Defaults to None.
        """
        self.type = llm_type
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_key = api_key
        # 收集模型和tokenizer, config
        self.models: dict = {}
        self.tokenizers: dict = {}
        self.device = "cuda"

    def chat(self, message: str) -> str:
        """ 模型的单轮对话

        Args:
            message (str): 用户输入

        Returns:
            str: 模型输出
        """
        type_ = self.type.name.split('_')[0]
        if type_.startswith('Qwen'):
            url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            body = {
                'model': self.type.value,
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
            return response.json(
            )['output']['choices'][0]['message']['content']
        elif type_.startswith('LocalQwen'):
            model_name = self.type.value
            if self.models.get(model_name, None) is None:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype="auto", device_map="auto")
                self.models[model_name] = model
            if self.tokenizers.get(model_name, None) is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.tokenizers[model_name] = tokenizer
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": message
            }]
            text = tokenizer.apply_chat_template(messages,
                                                 tokenize=False,
                                                 add_generation_prompt=True)
            model_inputs = tokenizer([text],
                                     return_tensors="pt").to(self.device)

            generated_ids = model.generate(model_inputs.input_ids,
                                           max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(
                    model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids,
                                              skip_special_tokens=True)[0]
            return response
        elif type_.startswith('LocalOneKE'):
            model_name = self.type.value
            if self.models.get(model_name, None) is None:
                # 4bit量化OneKE
                # quantization_config = BitsAndBytesConfig(
                #     load_in_4bit=True,
                #     llm_int8_threshold=6.0,
                #     llm_int8_has_fp16_weight=False,
                #     bnb_4bit_compute_dtype=torch.bfloat16,
                #     bnb_4bit_use_double_quant=True,
                #     bnb_4bit_quant_type="nf4",
                # )
                config = AutoConfig.from_pretrained(self.type.value,
                                                    trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.type.value,
                    config=config,
                    device_map="auto",
                    # quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                self.models[model_name] = model
            if self.tokenizers.get(model_name, None) is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.type.value, trust_remote_code=True)
                self.tokenizers[model_name] = tokenizer

            model = self.models[model_name]
            model.eval()
            tokenizer = self.tokenizers[model_name]

            system_prompt = '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n'
            sintruct = message
            sintruct = '[INST] ' + system_prompt + sintruct + '[/INST]'

            input_ids = tokenizer.encode(sintruct,
                                         return_tensors="pt").to(self.device)
            input_length = input_ids.size(1)
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=GenerationConfig(
                    max_length=1024,
                    max_new_tokens=512,
                    return_dict_in_generate=True),
                pad_token_id=tokenizer.eos_token_id)
            generation_output = generation_output.sequences[0]
            generation_output = generation_output[input_length:]
            response = tokenizer.decode(generation_output,
                                        skip_special_tokens=True)

            return response
