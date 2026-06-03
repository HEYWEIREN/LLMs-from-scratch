# 跨层 KV 共享

这份补充材料展示了把跨层 KV 共享与 KV cache 结合使用时可以节省多少内存。

&nbsp;
## 引言

在 [../04_gqa](../04_gqa) 中，我们讨论了 Grouped-Query Attention（GQA）：多个 query head 共享同一组 key 和 value head。跨层 KV 共享把一个相关思想应用到了 transformer 层之间。

与其在每一层都重新计算 key 和 value 投影，后续层会复用前面某一层的 K/V 张量。它们仍然会计算自己的 query，因此每一层仍然可以形成自己的注意力模式。主要的内存节省来自 cache 中需要存储的 K/V 张量更少。

这个思想也称为 cross-layer attention。Brandon *et al.* 的论文 [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981) 对此做了描述。Gemma 4 E2B 和 E4B 使用了相关的共享 KV-cache 方案，因此它是本章 GQA、MLA 和 SWA 示例之外的一个有用补充。

&nbsp;

<img src="gemma4-kv-sharing.webp" alt="Cross-layer KV sharing" width="800px" />

&nbsp;

在 [Gemma 4](../../ch05/17_gemma4) 中，KV 共享会与 GQA 或 MQA 以及滑动窗口注意力结合使用。对于本文件夹中的简化 GPT 示例，我们只实现跨层 KV 共享部分，让代码聚焦于主要机制。

这里使用的简化规则是：

1. 早期层计算并缓存自己的 K/V 张量。
2. 后续层复用最近一个早期生成层产生的 K/V 张量。
3. 所有层仍然计算自己的 query 投影。

这减少了会随上下文长度增长的 K/V cache 数量。代价是模型容量降低，因为有些层不再拥有自己的 K/V 投影。

&nbsp;
## KV 共享的内存节省

常规 KV-cache 内存计算如下：

bytes = batch_size × seqlen × head_dim × n_kv_heads × n_layers × 2 (K,V) × bytes_per_elem

使用跨层 KV 共享时，我们把 `n_layers` 替换为生成 K/V 的层数：

bytes = batch_size × seqlen × head_dim × n_kv_heads × n_kv_producing_layers × 2 (K,V) × bytes_per_elem

你可以使用本文件夹中的 [memory_estimator_kv_sharing.py](memory_estimator_kv_sharing.py) 脚本，把它应用到不同模型配置上：

```bash
# 类似 Gemma 4 E2B 的配置
uv run memory_estimator_kv_sharing.py \
  --context_length 131072 \
  --emb_dim 2048 \
  --n_heads 8 \
  --n_layers 35 \
  --n_kv_groups 8 \
  --n_kv_producing_layers 15 \
  --batch_size 1 \
  --dtype bf16

# 类似 Gemma 4 E4B 的配置
# uv run memory_estimator_kv_sharing.py \
#   --context_length 131072 \
#   --emb_dim 2560 \
#   --n_heads 8 \
#   --n_layers 42 \
#   --n_kv_groups 4 \
#   --n_kv_producing_layers 24 \
#   --batch_size 1 \
#   --dtype bf16

==== Config ====
context_length         : 131072
emb_dim                : 2048
n_heads                : 8
n_layers               : 35
n_kv_groups            : 8
n_kv_producing_layers  : 15
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 256
GQA n_kv_heads         : 1

==== KV-cache totals across all layers ====
MHA total KV cache        : 37.58 GB
GQA total KV cache        : 4.70 GB
MHA + KV sharing          : 16.11 GB
GQA + KV sharing          : 2.01 GB
Ratio (MHA / GQA+sharing) : 18.67x
Savings vs MHA            : 94.64%
```

这是一个类似 Gemma 4 E2B 的配置。35 层中包括 15 个生成 K/V 的层，其余层复用先前的 K/V 张量。对于类似 E4B 的配置，对应数字是 42 个总层数和 24 个生成 K/V 的层。

下面展示了类似 E2B 和 E4B 配置下的节省情况。为简单起见，这些图没有包含来自滑动窗口注意力的额外节省。

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e2b.webp" alt="KV-sharing memory savings for Gemma 4 E2B-like setup" width="800px" />

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e4b.webp" alt="KV-sharing memory savings for Gemma 4 E4B-like setup" width="800px" />

&nbsp;

可以通过下面的命令复现类似图：

```bash
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e2b
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e4b
```

&nbsp;
## KV 共享代码示例

本文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 脚本提供了动手示例，用于比较常规 MHA 和跨层 KV 共享变体。

查看实现细节最简单的方式，是比较 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) 的文件 diff。注释有意保持相似，这样 diff 会突出 KV 共享的改动。

注意，该模型没有训练，因此会生成无意义文本。不过你可以把它作为第 5-7 章中标准 GPT 模型的直接替代版本并进行训练。

此外，这个实现使用了[另一个补充章节](../03_kv-cache)中解释的 KV cache，因此内存节省会更明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768
```

```bash
uv run gpt_with_kv_sharing.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--n_kv_producing_layers 6
```

在这个小型 GPT 设置中，整个模型仍然包含相同的 feed-forward 层和输出头。主要内存差异在于有多少注意力层会在 cache 中存储 K/V 张量。
