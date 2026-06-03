# Mixture of Experts（MoE）

这份补充材料展示了使用 Mixture-of-Experts（MoE）层替代常规 feed-forward（FFN）层时，每个 token 可以节省多少内存。



&nbsp;
## 引言

MoE 的核心思想是：把 transformer block 中的每个 feed-forward 模块替换成多个专家层，而每个专家层本身也是一个 feed-forward 模块。也就是说，我们用多个 feed-forward block 替代单个 feed-forward block，如下图所示。



&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

Transformer block 内部的 feed-forward block（上图中的深灰色模块）通常包含模型总参数中的很大一部分。（注意，transformer block 以及其中的 feed-forward block 会在 LLM 中重复很多次；以 DeepSeek-V3 为例，重复了 61 次。）

因此，把*一个* feed-forward block 替换成*多个* feed-forward block（MoE 设置中的做法）会大幅增加模型总参数量。不过关键技巧是：我们不会为每个 token 使用（“激活”）所有专家。相反，router 会为每个 token 只选择一小部分专家。

由于每次只有少量专家处于激活状态，MoE 模块通常被称为*稀疏*模块，与总是使用完整参数集合的*稠密*模块相对。不过，MoE 带来的大量总参数会提升 LLM 容量，也就是训练时可以吸收更多知识；而稀疏性又能保持推理效率，因为不会同时使用所有参数。

例如，DeepSeek-V3 的每个 MoE 模块有 256 个专家，总参数量为 6710 亿。但推理时一次只激活 9 个专家（1 个共享专家加上 router 选出的 8 个）。这意味着每个 token 推理步骤只使用 370 亿参数，而不是全部 6710 亿参数。

DeepSeek-V3 的 MoE 设计中一个值得注意的特征是使用共享专家。共享专家会对每个 token 始终激活。这个想法并不新，早在 [2022 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 [2024 DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中就已经提出。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

（来自 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) 论文的一张带标注图。）

&nbsp;

拥有共享专家的好处最早在 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596)中被指出：相比没有共享专家，它能提升整体建模性能。这可能是因为常见或重复模式不需要被多个单独专家重复学习，从而让这些专家有更多空间学习更专门化的模式。

&nbsp;
## Mixture of Experts（MoE）的内存节省

MoE 模型中的内存节省主要来自激活存储和计算量的减少。在常规（稠密）feed-forward 层（FFN）中，每个 token 都会激活完整的中间维度。

相比之下，MoE 层会把每个 token 只路由到一小部分专家中，例如每个 token 使用 `num_experts` 中的 `top_k` 个专家。

使用 MoE 层时，每个 token 只有 `top_k` 个专家激活，因此相对于具有相同总容量的稠密 FFN，有效内存（和计算）大致按 `top_k / num_experts` 的比例缩放。


你可以使用本文件夹中的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本，把它应用到不同模型配置上，查看使用 MoE 替代 FFN 能节省多少内存（注意，这是针对单个 transformer block 的结果；如果要得到总节省量，需要乘以模型中的 transformer block 数）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

基于上面的结果可以看到，如果 FFN 的输入/输出维度（`emb_dim`）为 7,168，中间大小（`hidden_dim`）为 14,336，那么这一层约有 3.08 亿参数，并且所有这些参数都会在前向传播中激活。

现在，如果使用总参数量大致相同（约 3.08 亿）的 MoE 层，包含 8 个专家且每次激活 2 个专家，那么每次前向传播中只有约 7,700 万参数被激活。

此外，在专家总数固定时，专家越多，激活参数数量越低，“节省”越明显：

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



&nbsp;

可以通过下面的命令复现该图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE 代码示例

本文件夹中的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 脚本提供了动手示例，用于在 GPT 模型实现中比较常规 FFN 和 MoE 的内存占用。注意，这两个脚本都使用了第一张图所示的 [SwiGLU](https://arxiv.org/abs/2002.05202) feed-forward 模块（传统 GPT-2 使用 GELU）。

**注意：该模型没有训练，因此会生成无意义文本。你可以在补充材料 [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb) 中找到一个训练好的 MoE。**



首先，运行带常规 FFN 的模型：


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了和 MoE 公平比较，必须缩小专家大小。例如，如果使用 32 个专家，就需要设置 `--hidden_dim 32768/32`：


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

可以看到，稠密 feed-forward 层处理一个 token 约需 0.76 ms，并使用约 0.19 MB 激活内存（峰值接近 0.75 MB）。

稀疏 MoE 层只保留约 0.04 MB 内存（峰值 0.11 MB）。不过代价是计算时间大约翻倍。（这里有额外路由开销，而且我的实现也不一定最高效。）

整体生成过程在两种情况下的 GPU 内存峰值仍然约为 11.5 GB，因为两个版本加载了相同数量的权重参数，并且 KV cache 大小相同，这些因素在这里占主导。

无论如何，这展示了 MoE 的权衡：它把 FFN 内存降低约 4-5 倍，但大约让 feed-forward 计算时间翻倍。

注意，如果一次处理更多 token，例如 batch size 大于 1（这里为了代码简单没有使用 batch），节省会更明显。
