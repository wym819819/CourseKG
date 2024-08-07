# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/llm/__init__.py
# Description: 大模型接口

from .prompt import Prompt, ExamplePrompt
from .llm import LLM, VLLM, QwenAPI
from .visual_lm import MiniCPM
from .visual_prompt import MiniCPMPrompt
from .prompt_strategy import ExamplePromptStrategy
from .config import VisualConfig, LLMConfig
