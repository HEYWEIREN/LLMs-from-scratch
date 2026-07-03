# 第 5 章：无标签数据的预训练（Chapter 5: Pretraining on Unlabeled Data）

### 本章主代码（Main Chapter Code）

- [ch05.ipynb](ch05.ipynb) 包含本章中出现的所有代码
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，包含前面章节中的 `MultiHeadAttention` 模块和 `GPTModel` 类，我们在 [ch05.ipynb](ch05.ipynb) 中导入它们以预训练 GPT 模型
- [gpt_download.py](gpt_download.py) 包含用于下载预训练 GPT 模型权重的实用函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习题

### 可选代码（Optional Code）

- [gpt_train.py](gpt_train.py) 是一个独立的 Python 脚本文件，其中包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的用于训练 GPT 模型的代码（您可以将其视为总结本章的代码文件）
- [gpt_generate.py](gpt_generate.py) 是一个独立的 Python 脚本文件，其中包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的代码，用于加载和使用 OpenAI 的预训练模型权重