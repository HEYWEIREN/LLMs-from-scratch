# 第 4 章：从零实现用于生成文本的 GPT 模型

&nbsp;
## 主章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含本章的主代码。

&nbsp;
## 补充材料

- [02_performance-analysis](02_performance-analysis) 包含可选代码，用于分析主章节中实现的 GPT 模型性能
- [03_kv-cache](03_kv-cache) 实现 KV cache，用于在推理阶段加速文本生成
- [07_moe](07_moe) 讲解并实现 Mixture-of-Experts（MoE）
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) 包含一个分步骤指南，展示如何把 GPT 架构实现转换为 Llama 3.2，并加载 Meta AI 的预训练权重（如果你已经完成第 4 章，可以看看这些替代架构；也可以等读完第 5 章再看）


&nbsp;
## 注意力机制的替代方案

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa) 介绍 Grouped-Query Attention（GQA）。GQA 被大多数现代 LLM（Llama 4、gpt-oss、Qwen3、Gemma 3 等）用作常规 Multi-Head Attention（MHA）的替代方案
- [05_mla](05_mla) 介绍 Multi-Head Latent Attention（MLA）。MLA 被 DeepSeek V3 使用，也是一种常规 Multi-Head Attention（MHA）的替代方案
- [06_swa](06_swa) 介绍 Sliding Window Attention（SWA）。Gemma 3 等模型使用了这种机制
- [08_deltanet](08_deltanet) 讲解 Gated DeltaNet，这是一种流行的线性注意力变体（用于 Qwen3-Next 和 Kimi Linear）
- [10_kv-sharing](10_kv-sharing) 介绍跨层 KV 共享。Gemma 4 E2B 和 E4B 使用它来降低 KV-cache 内存占用


&nbsp;
## 更多

下面的视频提供了一场配套代码演示，作为本章部分内容的补充材料。

<br>
<br>

[![视频链接](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)
