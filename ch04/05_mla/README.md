# Multi-Head Latent Attention（MLA）

这份补充材料展示了使用 Multi-Head Latent Attention（MLA）替代常规 Multi-Head Attention（MHA）时可以节省多少内存。

&nbsp;
## 引言

在 [../04_gqa](../04_gqa) 中，我们讨论了 Grouped-Query Attention（GQA），它是 MHA 的一种计算效率折中方案。消融研究（例如[原始 GQA 论文](https://arxiv.org/abs/2305.13245)和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的实验）显示，从 LLM 建模性能来看，它与标准 MHA 相当。

现在，Multi-Head Latent Attention（MLA）提供了另一种节省内存的策略，并且特别适合与 KV cache 搭配使用。DeepSeek V2、V3 和 R1 使用了 MLA。MLA 不像 GQA 那样共享 key 和 value head，而是在把 key/value 存入 KV cache 之前，先把它们压缩到较低维空间。

推理时，这些压缩后的张量会先投影回原始大小，再被用于注意力计算，如下图所示。这会增加一次额外的矩阵乘法，但能降低内存使用。

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

（顺带一提，query 也会被压缩，但只在训练时压缩，推理时不压缩。）

另外，前面提到过，MLA 并不是 DeepSeek V3 才出现的；它的前身 [DeepSeek V2](https://arxiv.org/abs/2405.04434) 也使用并引入了 MLA。V2 论文还包含一些有意思的消融研究，可能解释了为什么 DeepSeek 团队选择 MLA 而不是 GQA（见下图）。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

如上图所示，GQA 看起来比 MHA 差，而 MLA 提供了比 MHA 更好的建模性能，这可能就是 DeepSeek 团队选择 MLA 而不是 GQA 的原因。（如果能看到 MLA 和 GQA 在“每 token KV Cache”节省上的对比，也会很有意思！）

总结本节，在进入下一个架构组件前，可以把 MLA 理解成一种巧妙技巧：它降低 KV cache 内存使用，同时在建模性能上甚至略优于 MHA。

&nbsp;
## MLA 的内存节省

内存节省主要体现在 KV 存储上。KV 存储大小可以用下面的公式计算：

bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem

相比之下，MHA 的 KV cache 内存计算如下：

bytes ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem

这意味着在 MLA 中，我们把 "embed_dim × 2 (K,V)" 减少为 "latent_dim"，因为我们只保存压缩后的 latent 表示，而不是完整 key 和 value 向量。



你可以使用本文件夹中的 [memory_estimator_mla.py](memory_estimator_mla.py) 脚本，把这个公式应用到不同模型配置上，查看使用 MLA 替代 MHA 能节省多少内存：

```bash
uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

注意，上面的压缩（`--emb_dim 2048 -> latent_dim 1024`）达到了与 GQA 类似的节省效果。实践中，压缩比例是一个需要仔细研究的超参数；如果 `latent_dim` 选得太小，会对建模性能产生负面影响，这类似于 GQA 中选择过多 `n_kv_groups` 的情况。

下图进一步展示了不同 `latent_dim` 值下，MLA 相对 MHA 随上下文长度变化带来的节省：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

可以通过 `uv run plot_memory_estimates_mla.py` 复现该图。



&nbsp;
## MLA 代码示例

本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 脚本提供了动手示例，用于在 GPT 模型实现中比较 MHA 和 MLA 的内存占用。

这里的 MLA 代码参考了 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) 实现。

注意，MLA 也可以和 [GQA](../04_gqa) 结合使用，但为了简单起见，这里没有这样做。（目前我也不知道有哪个知名 LLM 这样做。）

还要注意，该模型没有训练，因此会生成无意义文本。不过你可以把它作为第 5-7 章中标准 GPT 模型的直接替代版本并进行训练。

最后，这个实现使用了[另一个补充章节](../03_kv-cache)中解释的 KV cache，因此内存节省会更明显。

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

这里没有看到上图中那么大的节省，原因有两点：

1. 我使用了较小配置，目的是让模型能在合理时间内完成生成。
2. 更重要的是，这里观察的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（不过这是另一个单独分析主题）。
