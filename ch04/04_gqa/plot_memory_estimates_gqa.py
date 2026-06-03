# 版权所有 (c) Sebastian Raschka，遵循 Apache License 2.0（见 LICENSE.txt）。
# 《Build a Large Language Model From Scratch》的源代码
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

# 绘制不同 n_kv_groups 下 KV-cache 随上下文长度变化的曲线

import matplotlib.pyplot as plt

# 从 ./memory_estimator.py 导入
from memory_estimator_gqa import calc_kv_bytes_total, DTYPE_BYTES


def bytes_convert(n):
    gb = n / (1000 ** 3)
    return f"{gb:.2f}"


def savings_percent(total_mha, total_gqa):
    return (1.0 - (total_gqa / total_mha)) * 100.0


def plot_abs_kv_vs_context_multi_groups():
    n_heads = 24
    emb_dim = 2048
    n_layers = 48
    batch_size = 1
    dtype = "bf16"
    bytes_per_elem = DTYPE_BYTES[dtype]

    # x 轴（对数刻度）
    context_lengths = [
        256, 512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072
    ]

    mha_gb = []
    for L in context_lengths:
        total_mha = calc_kv_bytes_total(
            batch_size, L, emb_dim, n_heads,
            n_heads,  # MHA：n_kv_heads = n_heads
            n_layers, bytes_per_elem
        )
        mha_gb.append(float(bytes_convert(total_mha)))

    plt.figure()
    plt.plot(context_lengths, mha_gb, marker="o", label="MHA (KV total)")

    # 选定 n_kv_groups 的 GQA 曲线
    groups_list = [4, 8, 12, 24]
    for g in groups_list:
        n_kv_heads = n_heads // g
        gqa_gb = []
        for L in context_lengths:
            total_gqa = calc_kv_bytes_total(
                batch_size, L, emb_dim, n_heads,
                n_kv_heads, n_layers, bytes_per_elem
            )
            gqa_gb.append(float(bytes_convert(total_gqa)))

        # 相对于 MHA 的压缩率
        comp = (n_heads / n_kv_heads) if n_kv_heads > 0 else float("inf")
        plt.plot(context_lengths, gqa_gb, marker="o",
                 label=f"GQA (n_kv_groups={g}, {comp:,.1f}× compression)")

    plt.xscale("log")
    plt.xlabel("context_length (log scale)")
    plt.ylabel("Total KV cache (GB)")
    plt.title(
        "KV-cache vs Context Length — MHA vs GQA (multi-group)\n"
        "(n_heads=24, emb_dim=2048, n_layers=48, batch=1, dtype=bf16)",
        fontsize=8
    )
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("kv_bytes_vs_context_length.pdf")


if __name__ == "__main__":
    plot_abs_kv_vs_context_multi_groups()
