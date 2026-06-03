# Sliding Window Attention（SWA）

这份补充材料展示了使用 Sliding Window Attention（SWA）替代常规 Multi-Head Attention（MHA）时可以节省多少内存。



&nbsp;
## 引言

什么是滑动窗口注意力（SWA）？如果把常规自注意力看作一种*全局*注意力机制，因为每个序列元素都可以访问其他所有序列元素，那么 SWA 可以看作*局部*注意力，因为它限制了当前 query 位置周围的上下文大小。如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

如上图所示，每个 token 不再关注所有先前 token，而是只关注其位置附近一个固定大小的局部窗口。这种局部化注意力会显著降低 KV cache 的大小。

在本引言剩余部分，我们会结合 [Gemma 3](https://arxiv.org/abs/2503.19786) 讨论 SWA；Gemma 3 的从零实现位于 [../../ch05/12_gemma3](../../ch05/12_gemma3)。

滑动窗口注意力最初在 2020 年的 [LongFormer 论文](https://arxiv.org/abs/2004.05150)中提出，但这里关注 Google 的 Gemma 模型，是因为它们是非常优秀的开放权重模型，证明了滑动窗口注意力在近期能力较强的模型中确实可行。

[Gemma 2](https://arxiv.org/abs/2408.00118) 使用了局部（滑动窗口）和全局注意力层 1:1 组合的混合方案。每个 token 可以关注 4k token 的上下文窗口。采用这种 1:1 混合的原因是，它在效率和全局上下文建模之间取得了平衡；如果 LLM 只使用局部注意力，限制可能过强。

[Gemma 3](https://arxiv.org/abs/2503.19786) 进一步朝效率方向推进。它在滑动窗口层和全注意力层之间采用 5:1 比例，也就是说每五个局部注意力层之后有一个全局层。此外，滑动窗口大小也从 Gemma 2 的 4096 个 token 降低到 Gemma 3 的 1024 个 token。

有意思的是，Gemma 3 技术报告中的消融研究表明，这些变化对整体模型质量影响很小。换句话说，通过滑动窗口注意力获得的大量内存和计算节省，只带来了很小的建模性能损失。



&nbsp;
## Sliding Window Attention（SWA）的内存节省

内存节省主要体现在 KV 存储上。KV 存储大小可以用下面的公式计算：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

使用 SWA 时，我们把上式中的序列长度（seqlen）替换为窗口大小 W。因此，使用滑动窗口注意力时，KV cache 大小会按 "W / seqlen" 的比例下降。（为简单起见，这里假设每一层都使用滑动窗口注意力。）


你可以使用本文件夹中的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本，把它应用到不同模型配置上，查看使用 SWA 替代 MHA 能节省多少内存：

```bash
uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
GQA + SWA (Ratio: 5:1) : 0.78 GB
```

注意，Gemma 3 会把 SWA 和 GQA 结合使用。

下图进一步展示了不同上下文长度下，SWA 相对 MHA 带来的节省：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

&nbsp;

可以通过下面的命令复现这些图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA 代码示例

本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 脚本提供了动手示例，用于在 GPT 模型实现中比较 MHA 和 SWA 的内存占用。

注意，SWA 也可以和 MLA、GQA 结合使用（如前面提到的），但为了简单起见，这里没有这样做。

注意，该模型没有训练，因此会生成无意义文本。不过你可以把它作为第 5-7 章中标准 GPT 模型的直接替代版本并进行训练。

此外，这个实现使用了[另一个补充章节](../03_kv-cache)中解释的 KV cache，因此内存节省会更明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # 类似 Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

这里没有看到上图中那么大的节省，原因有两点：

1. 我使用了较小配置，目的是让模型能在合理时间内完成生成。
2. 更重要的是，这里观察的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（不过这是另一个单独分析主题）。
