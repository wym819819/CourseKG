# -*- coding: utf-8 -*-
# Create Date: 2024/07/11
# Author: wangtao <wangtao.cpu@gmail.com>
# File Name: coursekg/llm/config.py
# Description: 定义大模型配置

temperature = 0.6  # 取值越大softmax函数概率分布越平缓
top_p = 0.8  # 候选集仅保留累计概率之和大于等于top_p的顺序部分
top_k = 50  # 候选集仅保留概率最高的前top_k个，有限使用top_p
max_tokens = 3000  # 生成文本的最大长度
max_model_len = 9000  # 模型最长上下文长度

repetition_penalty = 1.05  # 重复token惩罚，被添加到softmax函数中
presence_penalty = 0  # 整个序列的重复度，被添加到损失函数中

tensor_parallel_size = 2  # 张量并行大小
