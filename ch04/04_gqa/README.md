# Grouped-Query Attention（GQA）

这份补充材料展示了使用 Grouped-Query Attention（GQA）替代常规 Multi-Head Attention（MHA）时可以节省多少内存。

&nbsp;
## 引言

近年来，Grouped-Query Attention（GQA）已经成为替代 Multi-Head Attention（MHA）的新标准选择，因为它在计算和参数上更高效。注意，GQA 并不是新概念，它可以追溯到 2023 年的论文 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)。甚至早期较大的 Llama 2 系列模型也使用了它。

下面是 GQA 的简要总结。MHA 中每个 head 都有自己的一组 key 和 value；而 GQA 为了降低内存使用，会把多个 head 分组，让同一组内的 head 共享相同的 key 和 value 投影。

例如，如下图进一步展示，如果有 3 个 key-value 组和 6 个注意力 head，那么 head 1 和 2 共享一组 key 和 value，head 3 和 4 共享另一组，head 5 和 6 也共享一组。

&nbsp;

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

这种 key 和 value 的共享减少了 key/value 的总计算量，从而降低内存使用并提高效率。

总结来说，GQA 的核心思想是：通过让多个 query head 共享 key/value head，减少 key 和 value head 的数量。这会（1）降低模型参数量，（2）在推理阶段减少 key/value 张量的内存带宽使用，因为 KV cache 中需要存储和读取的 key/value 更少。

虽然 GQA 主要是 MHA 的计算效率折中方案，但消融研究（例如[原始 GQA 论文](https://arxiv.org/abs/2305.13245)和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的实验）表明，从 LLM 建模性能来看，它与标准 MHA 相当。

不过，这个结论依赖于 key-value 组数的谨慎选择。在极端情况下，如果所有注意力 head 共享同一个 key-value 组，也就是 multi-query attention，内存占用会进一步大幅下降，但建模性能可能受损。（另一个极端是把 key-value 组数设为 query head 数，这就回到了标准多头注意力。）

&nbsp;
## GQA 的内存节省

内存节省主要体现在 KV 存储上。KV 存储大小可以用下面的公式计算：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

你可以使用本文件夹中的 [memory_estimator_gqa.py](memory_estimator_gqa.py) 脚本，把这个公式应用到不同模型配置上，查看使用 GQA 替代 MHA 能节省多少内存：

```bash
uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

下图进一步展示了不同 key-value 组大小下，GQA 相对 MHA 随上下文长度变化带来的节省：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

&nbsp;

可以通过 `uv run plot_memory_estimates_gqa.py` 复现该图。

&nbsp;
## GQA 代码示例

本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) 脚本提供了动手示例，用于在 GPT 模型实现中比较 MHA 和 GQA 的内存占用。

注意，GQA 也用于 [Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3) 和 [Qwen3](../../ch05/11_qwen3) 补充材料。不过为了简单起见，本文件夹中的代码脚本修改的是传统上没有使用 GQA 的 GPT 架构。

注意，该模型没有训练，因此会生成无意义文本。不过你可以把它作为第 5-7 章中标准 GPT 模型的直接替代版本并进行训练。

此外，这个实现使用了[另一个补充章节](../03_kv-cache)中解释的 KV cache，因此内存节省会更明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

这里没有看到上图中那么大的节省，原因有两点：

1. 我使用了较小配置，目的是让模型能在合理时间内完成生成。
2. 更重要的是，这里观察的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（不过这是另一个单独分析主题）。
