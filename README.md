# Build a Large Language Model (From Scratch)

本仓库包含用于开发、预训练（pretraining）和微调（finetuning）类 GPT LLM 的代码，也是《[Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D)》一书的官方代码仓库。

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

在 [*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv) 中，你将通过从零开始一步步编码，学习并理解大型语言模型（large language models, LLMs）如何从内部运作。本书会带你创建自己的 LLM，并用清晰的文字、图示和示例解释每个阶段。

本书中用于训练和开发小型但可用模型的教学方法，与构建 ChatGPT 等大规模基础模型（foundational models）背后的方法相呼应。此外，本书也包含加载更大规模预训练模型权重并进行 finetuning 的代码。

- 官方 [source code repository](https://github.com/rasbt/LLMs-from-scratch)
- [Manning 出版社的图书页面](http://mng.bz/orYv)
- [Amazon.com 图书页面](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

要下载本仓库副本，可以点击 [Download ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip)，或在终端中执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

（如果你是从 Manning 网站下载的代码包，建议也访问 GitHub 上的官方代码仓库 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)，以获取最新更新。）

<br>
<br>


# Table of Contents

请注意，`README.md` 是一个 Markdown（`.md`）文件。如果你是从 Manning 网站下载代码包并在本地电脑上查看，建议使用 Markdown 编辑器或预览器，以获得更好的阅读效果。如果还没有安装 Markdown 编辑器，[Ghostwriter](https://ghostwriter.kde.org) 是一个不错的免费选择。

你也可以在浏览器中通过 GitHub 查看此文件和其他文件：[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)，GitHub 会自动渲染 Markdown。

<br>
<br>


> **Tip:**
> 如果你需要安装 Python、Python packages，并配置代码环境，建议阅读 [setup](setup) 目录中的 [README.md](setup/README.md)。

<br>
<br>

[![Code tests Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Code tests Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![Code tests macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)

- [Troubleshooting Guide](./troubleshooting.md)


| 章节标题 | 主代码（便于快速访问） | 全部代码 + 补充材料 |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [Setup recommendations](setup) <br/>[How to best read this book](https://sebastianraschka.com/blog/2025/reading-books.html) | - | - |
| Ch 1: 理解大型语言模型 (Understanding Large Language Models) | 无代码 | - |
| Ch 2: 处理文本数据 (Working with Text Data) | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (summary)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb) | [./ch02](./ch02) |
| Ch 3: 编写注意力机制 (Coding Attention Mechanisms) | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (summary) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb) | [./ch03](./ch03) |
| Ch 4: 从零实现 GPT 模型 (Implementing a GPT Model from Scratch) | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (summary)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04) |
| Ch 5: 在无标签数据上预训练 (Pretraining on Unlabeled Data) | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (summary) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (summary) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05) |
| Ch 6: 文本分类微调 (Finetuning for Text Classification) | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06) |
| Ch 7: 指令跟随微调 (Finetuning to Follow Instructions) | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (summary)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (summary)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07) |
| Appendix A: PyTorch 简介 (Introduction to PyTorch) | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| Appendix B: 参考资料与延伸阅读 (References and Further Reading) | 无代码 | [./appendix-B](./appendix-B) |
| Appendix C: 练习答案 (Exercise Solutions) | - [练习答案列表](appendix-C) | [./appendix-C](./appendix-C) |
| Appendix D: 为训练循环添加 Bells and Whistles | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb) | [./appendix-D](./appendix-D) |
| Appendix E: 使用 LoRA 进行参数高效微调 (Parameter-efficient Finetuning with LoRA) | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb) | [./appendix-E](./appendix-E) |

<br>
&nbsp;

下面的 mental model 总结了本书覆盖的内容。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">


<br>
&nbsp;

## Prerequisites

最重要的先修基础是扎实的 Python 编程能力。
具备这一基础后，你就能更好地探索 LLMs 的世界，
并理解本书中介绍的概念和代码示例。

如果你有一些深度神经网络（deep neural networks）经验，某些概念可能会更容易理解，因为 LLMs 构建在这些架构之上。

本书使用 PyTorch 从零实现代码，不依赖任何外部 LLM libraries。熟练掌握 PyTorch 并不是硬性先修要求，但熟悉 PyTorch basics 会很有帮助。如果你刚开始学习 PyTorch，Appendix A 提供了一个简洁的 PyTorch 入门。你也可以参考我的书 [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/)，学习相关基础。



<br>
&nbsp;

## Hardware Requirements

本书主章节中的代码设计目标是在普通笔记本电脑上用合理时间运行，不需要专门硬件。这样的设计能让更多读者参与学习。另外，如果 GPU 可用，代码会自动使用 GPU。（更多建议请参阅 [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) 文档。）


&nbsp;
## Video Course

这里有一个 [17 小时 15 分钟的配套视频课程](https://www.manning.com/livevideo/master-and-build-large-language-models)，我会按章节逐步编写本书代码。课程按照与图书结构一致的章节和小节组织，因此既可以作为本书的独立替代学习资源，也可以作为配套的 code-along 资源。

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>


&nbsp;


## Companion Book / Sequel

[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 是一本独立图书，但也可以看作 *Build A Large Language Model (From Scratch)* 的续作。

它从一个 pretrained model 出发，实现多种 reasoning 方法，包括 inference-time scaling、reinforcement learning 和 distillation，以提升模型的 reasoning capabilities。

与 *Build A Large Language Model (From Scratch)* 类似，[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 采用 hands-on 方式，从零实现这些方法。

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="120px"></a>

- Amazon 链接 (TBD)
- [Manning 链接](https://mng.bz/lZ5B)
- [GitHub repository](https://github.com/rasbt/reasoning-from-scratch)

<br>

&nbsp;
## Exercises

本书每章都包含若干练习。答案汇总在 Appendix C 中，对应的代码 notebooks 位于本仓库的主章节文件夹中，例如 [./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)。

除了代码练习，你还可以从 Manning 网站免费下载一本 170 页的 PDF：[Test Yourself On Build a Large Language Model (From Scratch)](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)。其中每章大约包含 30 道 quiz questions 和 solutions，帮助你检验理解程度。

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>

&nbsp;
## Bonus Material

若干文件夹中包含供感兴趣读者使用的可选 bonus materials：
- **Setup**
  - [Python Setup Tips](setup/01_optional-python-setup-preferences)
  - [Installing Python Packages and Libraries Used in This Book](setup/02_installing-python-libraries)
  - [Docker Environment Setup Guide](setup/03_optional-docker-environment)

- **Chapter 2: Working With Text Data**
  - [从零实现 Byte Pair Encoding (BPE) Tokenizer](ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb)
  - [比较多种 Byte Pair Encoding (BPE) 实现](ch02/02_bonus_bytepair-encoder)
  - [理解 Embedding Layers 与 Linear Layers 的区别](ch02/03_bonus_embedding-vs-matmul)
  - [用简单数字建立 Dataloader Intuition](ch02/04_bonus_dataloader-intuition)

- **Chapter 3: Coding Attention Mechanisms**
  - [比较高效的 Multi-Head Attention 实现](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [理解 PyTorch Buffers](ch03/03_understanding-buffers/understanding-buffers.ipynb)

- **Chapter 4: Implementing a GPT Model From Scratch**
  - [FLOPs Analysis](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV Cache](ch04/03_kv-cache)
  - [Attention Alternatives](ch04/#attention-alternatives)
    - [Grouped-Query Attention](ch04/04_gqa)
    - [Multi-Head Latent Attention](ch04/05_mla)
    - [Sliding Window Attention](ch04/06_swa)
    - [Gated DeltaNet](ch04/08_deltanet)
    - [Cross-Layer KV Sharing](ch04/10_kv-sharing)
  - [Mixture-of-Experts (MoE)](ch04/07_moe)

- **Chapter 5: Pretraining on Unlabeled Data**
  - [Alternative Weight Loading Methods](ch05/02_alternative_weight_loading/)
  - [在 Project Gutenberg Dataset 上预训练 GPT](ch05/03_bonus_pretraining_on_gutenberg)
  - [为 Training Loop 添加 Bells and Whistles](ch05/04_learning_rate_schedulers)
  - [为 Pretraining 优化 Hyperparameters](ch05/05_bonus_hparam_tuning)
  - [构建用于交互 Pretrained LLM 的 User Interface](ch05/06_user_interface)
  - [将 GPT 转换为 Llama](ch05/07_gpt_to_llama)
  - [Memory-efficient Model Weight Loading](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [为 Tiktoken BPE Tokenizer 扩展新 Tokens](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [用于更快 LLM Training 的 PyTorch Performance Tips](ch05/10_llm-training-speed)
  - [LLM Architectures](ch05/#llm-architectures-from-scratch)
    - [Llama 3.2 From Scratch](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
    - [Qwen3 Dense and Mixture-of-Experts (MoE) From Scratch](ch05/11_qwen3/)
    - [Gemma 3 From Scratch](ch05/12_gemma3/)
    - [Olmo 3 From Scratch](ch05/13_olmo3/)
    - [Tiny Aya From Scratch](ch05/15_tiny-aya/)
    - [Qwen3.5 From Scratch](ch05/16_qwen3.5/)
    - [Gemma 4 E2B and E4B From Scratch](ch05/17_gemma4/)
  - [将其他 LLMs 作为 Chapter 5 的 Drop-In Replacement（例如 Llama 3、Qwen 3）](ch05/14_ch05_with_other_llms/)
- **Chapter 6: Finetuning for classification**
  - [微调不同层和使用更大模型的 Additional Experiments](ch06/02_bonus_additional-experiments)
  - [在 50k IMDb Movie Review Dataset 上微调不同模型](ch06/03_bonus_imdb-classification)
  - [构建用于交互 GPT-based Spam Classifier 的 User Interface](ch06/04_user_interface)
- **Chapter 7: Finetuning to follow instructions**
  - [用于查找 Near Duplicates 和创建 Passive Voice Entries 的 Dataset Utilities](ch07/02_dataset-utilities)
  - [使用 OpenAI API 和 Ollama 评估 Instruction Responses](ch07/03_model-evaluation)
  - [生成 Instruction Finetuning 数据集](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [改进 Instruction Finetuning 数据集](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [使用 Llama 3.1 70B 和 Ollama 生成 Preference Dataset](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [用于 LLM Alignment 的 Direct Preference Optimization (DPO)](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [构建用于交互 Instruction-Finetuned GPT Model 的 User Interface](ch07/06_user_interface)

更多 bonus material 来自 [Reasoning From Scratch](https://github.com/rasbt/reasoning-from-scratch) 仓库：

- **Qwen3 (From Scratch) Basics**
  - [Qwen3 Source Code Walkthrough](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb)
  - [Optimized Qwen3](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM)

- **Evaluation**
  - [Verifier-Based Evaluation (MATH-500)](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03)
  - [Multiple-Choice Evaluation (MMLU)](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/02_mmlu)
  - [LLM Leaderboard Evaluation](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/03_leaderboards)
  - [LLM-as-a-Judge Evaluation](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/04_llm-judge)
- **Inference Scaling**
  - [Self-Consistency](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/01_main-chapter-code/ch04_main.ipynb)
  - [Self-Refinement](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/01_main-chapter-code/ch05_main.ipynb)

- **Reinforcement Learning** (RL)
  - [RLVR with GRPO From Scratch](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_main.ipynb)


<br>
&nbsp;

## Questions, Feedback, and Contributing to This Repository


欢迎各种反馈，最好通过 [Manning Forum](https://livebook.manning.com/forum?product=raschka&page=1) 或 [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions) 分享。同样，如果你有任何问题，或者只是想和其他读者交流想法，也欢迎在论坛中发帖。

请注意，由于本仓库包含与纸质书对应的代码，我目前无法接受会扩展主章节代码内容的贡献，因为这会造成与实体书内容的偏差。保持一致有助于确保所有读者获得顺畅的学习体验。


&nbsp;
## Citation

如果你觉得本书或代码对你的研究有帮助，请考虑引用它。

Chicago-style citation:

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX entry:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
