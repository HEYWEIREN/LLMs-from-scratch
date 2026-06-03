# 用于线性注意力的 Gated DeltaNet

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 和 [Kimi Linear](https://arxiv.org/abs/2510.26692) 提出了混合 transformer，它们实现了注意力机制的替代方案，相对于上下文长度按线性而不是二次方扩展。

Qwen3-Next 和 Kimi Linear 都使用 3:1 比例，也就是每三个采用线性 Gated DeltaNet 变体的 transformer block，就有一个使用完整注意力的 block，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next versus Kimi Linear">



&nbsp;

## 引言和概览

Gated DeltaNet 是一种线性注意力变体，灵感来自循环神经网络，并包含 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文中的门控机制。从某种意义上说，Gated DeltaNet 是带有 Mamba 风格门控的 DeltaNet，而 DeltaNet 是一种线性注意力机制。

Kimi Linear 通过 Kimi Delta Attention（KDA）机制修改了 Qwen3-Next 的线性注意力机制；KDA 本质上是 Gated DeltaNet 的一个改进版本。Qwen3-Next 使用标量 gate（每个注意力 head 一个值）控制记忆衰减率，而 Kimi Linear 将其替换为每个特征维度的通道级门控。作者认为，这能更细粒度地控制记忆，从而提升长上下文推理能力。

此外，对于完整注意力层，Kimi Linear 将 Qwen3-Next 的 gated attention 层（本质上是带输出门控的标准多头注意力层）替换为 Multi-Head Latent Attention（MLA）。这与我们前面在 DeepSeek V3/R1 部分讨论过的 MLA 机制相同，只是额外加了一个 gate。（回顾一下，MLA 会压缩 key/value 空间以降低 KV cache 大小。）

Kimi Linear 中的 MLA 没有使用 gate，这是有意设计的，方便作者更直接地把该架构与标准 MLA 比较；不过他们[表示](https://x.com/yzhang_cs/status/1984631714464088563)未来计划加入它。

由于我们已经在 [../05_mla](../05_mla) 中实现过 MLA，这份补充材料主要关注 Gated DeltaNet。


&nbsp;
## Gated Attention

在进入 Gated DeltaNet 本身之前，先简单谈谈 gate。正如前一张图中 Qwen3-Next 架构上半部分所示，Qwen3-Next 使用了“gated attention”。它本质上是在常规完整注意力上额外加了一个 sigmoid gate。

下面为了说明，把这个门控作为一个简单改动加到第 3 章的 `MultiHeadAttention` 代码中：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### 新增：添加 gate
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### 新增：添加 gate
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### 新增：添加 gate
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



可以看到，在照常计算注意力之后，模型会根据同一个输入生成一个单独的门控信号，对它应用 sigmoid，使其位于 0 到 1 之间，然后将它与注意力输出相乘。这允许模型动态放大或缩小某些特征。Qwen3-Next 开发者[表示](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)，这有助于训练稳定性：

> [...] the attention output gating mechanism helps eliminate issues like Attention Sink and Massive Activation, ensuring numerical stability across the model.


&nbsp;
## Gated DeltaNet

那么，Gated DeltaNet 是什么？Gated DeltaNet（*Gated Delta Network* 的缩写）是 Qwen3-Next 的线性注意力层，目的是作为标准 softmax attention 的替代方案。如前所述，它来自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文。

Gated DeltaNet 最初被提出为 Mamba2 的改进版本，它把 Mamba2 的门控衰减机制与 delta rule 结合起来。

Mamba 是一种状态空间模型（transformer 的替代方案），这是一个很大的主题，值得以后单独介绍。

delta rule 部分指的是计算新 value 和预测 value 之间的差值（delta，Δ），并用它更新一个作为记忆状态的隐藏状态（后面会进一步说明）。

（旁注：熟悉经典机器学习文献的读者可以把它看成类似受生物学启发的 Hebbian learning："Cells that fire together wire together." 它基本上是感知机更新规则和基于梯度下降学习的前身，但不使用监督信号。）

Gated DeltaNet 有一个与前面 gated attention 类似的 gate，只不过它使用 SiLU 而不是 logistic sigmoid 激活，如下图所示。（选择 SiLU 很可能是为了相比标准 sigmoid 改善梯度流和稳定性。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>

不过，如上图所示，Gated DeltaNet 中的 “gated” 还指几个额外 gate：

- `α`（decay gate）控制记忆随时间衰减或重置的速度，
- `β`（update gate）控制新输入对状态的修改强度。

在代码中，上图所示 Gated DeltaNet 的简化版本（不含卷积混合）可以如下实现；代码参考了 Qwen3 团队的[官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835)。

（注意，有些实现会把 decay gate 称为 `gk`（step k 的 gate），其中 `exp(gk)` 对应论文中的 $lpha_t$。为了让这个关系更明确，下面的代码片段把 log 空间的 gate `alpha_log` 与指数化后的衰减 `alpha` 分开。）


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### 新增：delta rule 和输出门控所需的 gate
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # 注意：衰减 gate alpha 对应
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # 也可以把它实现为
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # 但这里为了可解释性，并为了模拟官方实现，
        # 将 bias 单独拆出来

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### 新增：计算 delta rule 的 gate
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # 新增

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)
        gate = gate.transpose(1, 2)  # 新增

        ####################################################
        ### 新增：用于 delta rule 的类 QKNorm 归一化
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### 新增：Gated delta rule 更新
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### 新增：应用 RMSNorm 和 SiLU gate
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（注意，为了简单起见，我省略了 Qwen3-Next 和 Kimi Linear 使用的卷积混合，这样代码更可读，也能把重点放在循环状态方面。）

可以看到，上面的代码和标准（或 gated）attention 有很多差异。

在 gated attention 中，模型会在所有 token 之间计算普通注意力（每个 token 都会关注或查看其他 token）。然后，在得到注意力输出后，一个 gate（sigmoid）决定保留多少输出。关键点是，它仍然是常规 scaled-dot product attention，因此会随上下文长度按二次方扩展。

回顾一下，scaled-dot product attention 计算为 softmax(QKᵀ)V，其中 Q 和 K 是 *n*×*d* 矩阵，*n* 是输入 token 数，*d* 是嵌入维度。因此 QKᵀ 会产生一个 *n*×*n* 的注意力矩阵，再乘以一个 *n*×*d* 的 value 矩阵 V：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="Quadratic attention" width=500px />

在 Gated DeltaNet 中，没有 *n*×*n* 注意力矩阵。相反，模型会一个 token 一个 token 地处理。它维护一个运行中的记忆（状态），每当新 token 进入时都会更新该记忆。这对应下面的实现，其中 `S` 是在每个时间步 *t* 递归更新的状态。

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

这些 gate 控制记忆如何变化：

- α（`alpha`）调节旧记忆被遗忘（衰减）的程度。

- β（`beta`）调节当前时间步 *t* 的 token 对记忆的更新强度。

（最终输出 gate 在上面的代码片段中没有展示，它类似于 gated attention，用来控制保留多少输出。）

因此，从某种意义上说，Gated DeltaNet 中的状态更新类似于循环神经网络（RNN）的工作方式。它的优势是随上下文长度线性扩展（通过 for 循环），而不是二次方扩展。

这种循环状态更新的缺点是：相比常规（或 gated）attention，它牺牲了来自完整两两注意力的全局上下文建模能力。

Gated DeltaNet 在一定程度上仍然可以捕获上下文，但必须经过记忆状态（*S*）这个瓶颈。该记忆是固定大小的，因此更高效，但它会像 RNN 一样把过去上下文压缩进单个隐藏状态。

这也是为什么 Qwen3-Next 和 Kimi Linear 架构不会用 DeltaNet 层替换所有注意力层，而是使用前面提到的 3:1 比例。

&nbsp;
## DeltaNet 的内存节省

上一节讨论了 DeltaNet 相比完整注意力的优势：相对于上下文长度，它的计算复杂度是线性的，而不是二次方的。

除了线性计算复杂度，DeltaNet 的另一个重要优势是内存节省，因为 DeltaNet 模块不会让 KV cache 增长。（关于 KV cache 的更多信息，见 [../03_kv-cache](../03_kv-cache)）。如前所述，它们会维护一个固定大小的循环状态，因此内存不会随上下文长度增长。

对于常规多头注意力（MHA）层，KV cache 大小可以按如下方式计算：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

其中乘以 2 是因为缓存中同时存储 key 和 value。

对于上面实现的简化 DeltaNet 版本，有：


```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

注意，`KV_cache_DeltaNet` 的内存大小不依赖上下文长度（`n_tokens`）。此外，我们只存储记忆状态 S，而不是分别存储 key 和 value，因此 `2 × bytes` 变成了 `bytes`。不过也要注意，这里现在出现了一个二次项 `d_head × d_head`，它来自状态：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

但这通常不需要担心，因为 head 维度一般相对较小。例如，Qwen3-Next 中是 128。

带卷积混合的完整版本会更复杂，还涉及 kernel size 等因素；但上面的公式已经足以说明 Gated DeltaNet 背后的主要趋势和动机。

可以通过下面的辅助脚本可视化不同上下文长度下的内存估计和节省：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

注意，上面会把 `head_dim` 计算为 `emb_dim / n_heads`，即 2048 / 16 = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet scaling" width=500px>
