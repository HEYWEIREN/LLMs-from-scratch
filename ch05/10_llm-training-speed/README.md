# 加快 LLM 训练速度的 PyTorch 性能技巧（PyTorch Performance Tips for Faster LLM Training）



请注意，本书是出于教育目的而编写的，这意味着原始代码有意保持简单。这是为了提高可读性并确保不同硬件（包括 CPU 和 GPU）之间的兼容性。但是，您可能对一些更高级的 PyTorch 和 GPU 功能感到好奇，以使 LLM 训练更加高效。

该文件夹包含三个代码文件，演示了第 5 章中介绍的 LLM 和训练功能的性能优化：

1.[`00_orig.py`](00_orig.py)：CPU和单GPU训练的原始第5章代码。
   ➤ 运行方式：`python 00_orig.py`

2.[`01_opt_single_gpu.py`](01_opt_single_gpu.py)：针对单GPU训练的优化版本。
   ➤ 运行方式：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py)：使用分布式数据并行（DDP）的多GPU训练的优化版本。
   ➤ 运行方式：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`
   （**注意：** 为了保持与 `01_opt_single_gpu.py` 相比的最小变化，此脚本仅通过 `torchrun` 支持多处理，如上所示。这意味着通过 `python 02_opt_multi_gpu_ddp.py` **不** 支持多 GPU 支持）

**请注意，这些修改将训练速度从每秒 12,525 个token（单个 A100）提高到每秒 142,156 个token（单个 A100）和每秒 419,259 个token（4 个 A100）。**

我计划在将来的某个时候在更详细的文章中扩展这些差异。目前，查看代码中添加了哪些改进的最简单方法是在 Visual Studio Code 中打开文件并通过“比较所选”功能查看差异。

![VS比较](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch 提示](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)


 
## 单 GPU 速度比较（Single GPU speed comparisons）

正如上面提到的，我计划在未来详细阐述这些变化。目前，本节包含每次修改的token/秒的简单性能概述。所有实验均在 A100 GPU 上运行。

 
### 基线（Baseline）

请注意，`00_orig.py` 服务器作为基线，不包含任何重大修改，并使用第 5 章中的代码，除了以下内容之外：

- 上下文长度增加了 4 倍（这解释了与第 5 章相比 `00_orig.py` 的内存占用相对较大）；
- 4倍批量大小变化（`00_orig.py`内存占用相对较大的另一个原因）；
- 更大的公共领域书籍以增加训练数据大小。超参数对于最小化损失和减少过度拟合并不是非常优化，并且LLM在最后生成的文本可能不是超级复杂；然而，这并不重要，因为主要的要点是 `tok/sec` 指标，它在这里充当速度参考（越高越好）。



```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```



请注意，`01_opt_single_gpu.py` 包含下面按顺序列出的所有修改。

比较始终基于上一节中第一个 epoch 之后的平均 tok/sec 和分配的内存。

 
### 1. 动态创建因果掩码（1. Create causal mask on the fly）

- 这不是保存因果掩码，而是动态创建因果掩码以减少内存使用（这里它的影响很小，但它可以在长上下文大小模型中累加，例如支持 131k-input-tokens 的 Llama 3.2）

之前：
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

之后：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

 
### 2. 使用张量核心（2. Use  tensor cores）

- 使用张量核心（仅适用于 Ampere GPU，如 A100 及更新版本）

之前：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

 
### 3. 融合 AdamW 优化器（Fused AdamW optimizer）

- 通过设置 `fused=True` 使用 `AdamW` 的融合内核

之前：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

 
### 4. 数据加载器中的固定内存（4. Pinned memory in the data loader）

- 在数据加载器中使用 `pin_memory=True` 来预分配和重用 GPU 内存

之前：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

 
### 5. 使用 bfloat16 精度（5. Using bfloat16 precision）

- 从 32 位浮点数切换到 16 位脑浮点数 (bfloat16) 精度（有关此主题的更多信息，请参阅我的[此处的文章](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）

之前：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

 
### 6. 用 PyTorch 类替换从零实现代码（6. Replacing from-scratch code by PyTorch classes）

- 用 PyTorch 的本机实现替换了 LayerNorm 和 GeLU 从零实现

之前：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

之后：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

 
### 7. 使用 FlashAttention（7. Using FlashAttention）

- 将 PyTorch 的自注意力函数与 FlashAttention 结合使用，而不是我们从零实现的多头注意力实现。


之前：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

之后：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

 
### 8. 使用 `torch.compile`（Using `pytorch.compile`）

- 使用 `torch.compile(model)`。最初几次迭代通常较慢，之后才会加速。由于 `Avg tok/sec` 的平均值包含较慢的首次迭代，这里改用第 1 个 epoch 结束时的 `Step tok/sec`。


之前：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

之后：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows 注意**

- 在 Windows 上编译可能会很棘手
- `torch.compile()` 使用 Inductor，它 JIT 编译内核并需要一个可用的 C/C++ 工具链
- 对于 CUDA，Inductor 还依赖于 Triton，可通过社区包 `triton-windows` 获得
  - 如果您看到 `cl not found`，[使用“C++ 工作负载”](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) 安装 Visual Studio 构建工具，并从“x64 Native Tools”提示符运行 Python
  - 如果您看到带有 CUDA 的 `triton not found`，请安装 `triton-windows`（例如 `uv pip install "triton-windows<3.4"`）。
- 对于 CPU，读者进一步建议遵循此[Windows 的 PyTorch 电感指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
  - 这里，安装Visual Studio 2022时一定要安装英文语言包，以避免出现UTF-8错误
  - 另外，请注意，代码需要通过“Visual Studio 2022 开发人员命令提示符”而不是笔记本来运行
- 如果这个设置被证明很棘手，您可以跳过编译； **编译是可选的，所有代码示例在没有它的情况下都可以正常工作**

---

 
### 9. 词汇表填充（Vocabulary padding）

- 这里将词汇表大小从 50,257 略微增加到 50,304，即最近的 64 的倍数。这项技巧来自 Andrej Karpathy 的建议，依据是更适合张量核心的形状通常能提升计算效率。可进一步参考 [NVIDIA 的张量形状指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)以及 2019 年的 [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)。

之前：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

之后：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

 
### 10. 增加批量大小（Increasing the batch size）

- 最后，我们将批量大小增加到 GPU 支持的最大 2 次方

之前：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

之后：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`


 
## 多 GPU 速度比较（Multi-GPU speed comparisons）

这可能不是一个完全公平的比较，因为我们现在使用 4 个 GPU，而不是 1 个，但使用分布式数据并行性，如果训练不受有限 GPU 内存的瓶颈，则可以使用最快的多 GPU 技术，当然可以带来显着的加速：

之前（单 GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

之后（4 个 GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`
