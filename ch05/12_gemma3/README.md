# 从零实现 Gemma 3 270M（Gemma 3 270M From Scratch）

此文件夹中的此 [standalone-gemma3.ipynb](standalone-gemma3.ipynb) Jupyter 笔记本包含 Gemma 3 270M 的从零实现。它需要大约 2 GB 的 RAM 才能运行。

替代方案 [standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb) 笔记本添加了 KV 缓存，以实现更好的运行时性能（但增加了更多代码复杂性）。要了解有关 KV 缓存的更多信息，请参阅我的 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 文章。

|型号|模式|硬件|token/秒 | GPU 内存 (VRAM) |
| ----------------- | ----------------- | ---------------- | ---------- | ----------------- |
| Gemma3型号 270M |常规| Mac Mini M4 CPU | 8 | - |
| Gemma3型号 270M |常规编译| Mac Mini M4 CPU | 9 | - |
| Gemma3型号 270M | KV缓存| Mac Mini M4 CPU | 130 | 130 - |
| Gemma3型号 270M | KV缓存编译| Mac Mini M4 CPU | 224 | 224 - |
|                   |                   |                 |            |                   |
| Gemma3型号 270M |常规| Mac Mini M4 GPU | 16 | 16 - |
| Gemma3型号 270M |常规编译| Mac Mini M4 GPU |错误 | - |
| Gemma3型号 270M | KV缓存| Mac Mini M4 GPU | 23 | 23 - |
| Gemma3型号 270M | KV缓存编译| Mac Mini M4 GPU |错误 | - |
|                   |                   |                 |            |                   |
| Gemma3型号 270M |常规| Nvidia A100 GPU | 28 | 28 1.84 GB | 1.84 GB
| Gemma3型号 270M |常规编译| Nvidia A100 GPU | 128 | 128 2.12 GB | 2.12 GB
| Gemma3型号 270M | KV缓存| Nvidia A100 GPU | 26 | 26 1.77 GB | 1.77 GB
| Gemma3型号 270M | KV缓存编译| Nvidia A100 GPU | 99 | 99 2.12 GB | 2.12 GB


下面是与Qwen3 0.6B作为参考模型的并排比较；如果你对Qwen3 0.6B独立笔记本感兴趣，可以找到它[这里](../11_qwen3)。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

要了解有关架构差异的更多信息并阅读与其他架构的比较，请参阅我的[大型 LLM 架构比较：从 DeepSeek-V3 到 Kimi K2：现代 LLM 架构设计概览](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 文章。