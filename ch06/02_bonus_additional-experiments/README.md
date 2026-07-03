# 额外的分类微调实验（Additional Classification Finetuning Experiments）

下表增加了一组实验，用来回答关于不同设计选择的额外问题。第一行使用与主章节相同的设置，作为参考基线。
例如：

- 比较第 1 行和第 2 行，可以回答：“训练最后一个 token 与第一个 token 的性能差异是什么？”
- 比较第 1 行和第 3 行，可以回答：“只训练最后一层而不是最后一个 block，性能差异是什么？”
- 依此类推。

&nbsp;

|      | Model              | Weights    | Trainable token position | Trainable layers | Context length                                         | Training acc | Validation acc | Test acc | Training time | CPU/GPU |
| ---- | ------------------ | ---------- | ------------------------ | ---------------- | ------------------------------------------------------ | ------------ | -------------- | -------- | ------------- | ------- |
| 1    | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 2    | gpt2-small (124M)  | pretrained | first                    | last_block       | longest train ex. (120)                                | 78.46%       | 80.54%         | 75.00%   | 0.28 min      | A100    |
| 3    | gpt2-small (124M)  | pretrained | last                     | last_layer       | longest train ex. (120)                                | 78.65%       | 79.87%         | 72.00%   | 0.25 min      | A100    |
| 4    | gpt2-small (124M)  | pretrained | last                     | last_two_blocks  | longest train ex. (120)                                | 98.85%       | 98.66%         | 98.33%   | 0.33 min      | A100    |
| 5    | gpt2-small (124M)  | pretrained | last                     | all              | longest train ex. (120)                                | 99.62%       | 96.64%         | 96.67%   | 0.69 min      | A100    |
| 6    | gpt2-medium (355M) | pretrained | last                     | last_block       | longest train ex. (120)                                | 87.50%       | 91.28%         | 84.67%   | 0.75 min      | A100    |
| 7    | gpt2-large (774M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.52%       | 98.66%         | 96.67%   | 1.50 min      | A100    |
| 8    | gpt2-xl (1558M)    | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.81%       | 99.81%         | 98.33%   | 2.83 min      | A100    |
| 9    | gpt2-xl (1558M)    | pretrained | last                     | all              | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.67%   | 8.12 min      | A100    |
| 10   | gpt2-small (124M)  | random     | last                     | all              | longest train ex. (120)                                | 100.00%      | 96.64%         | 93.67%   | 0.69 min      | A100    |
| 11   | gpt2-small (124M)  | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 97.32%         | 96.67%   | 0.75 min      | A100    |
| 12   | gpt2-xl (1558M)    | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.33%   | 5.79 min      | A100    |
| 13   | gpt2-small (124M)  | pretrained | last                     | last_block       | context length (1024)                                  | 83.08%       | 87.92%         | 78.33%   | 2.46 min      | A100    |
| 14   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 1)                    | 100.00%      | 98.66%         | 98.00%   | 1.75 min      | A100    |
| 15   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 8)                    | 99.33%       | 98.66%         | 98.33%   | 1.70 min      | A100    |
| 16   | gpt2-small (124M)  | pretrained | last                     | last_block       | flexible (last non-padding position)                   | 99.42%       | 98.66%         | 98.33%   | 0.30 min      | A100    |
| 17   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120); but no causal mask            | 99.23%       | 98.66%         | 95.33%   | 0.29 min      | A100    |
| 18   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120) and `ignore_index` for padding | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 19   | gpt2-small (124M)  | pretrained | last + pooled embeddings | last_block       | longest train ex. (120)                                | 97.79%       | 99.33%         | 96.33%   | 0.32 min      | A100    |

&nbsp;

### 使用方式（Usage）

可以使用下面的代码复现实验：

- Row 1: `python additional_experiments.py`
- Row 2: `python additional_experiments.py --trainable_token_pos first`
- Row 3: `python additional_experiments.py --trainable_layers last_layer`
- Row 4: `python additional_experiments.py --trainable_layers last_two_blocks`
- Row 5: `python additional_experiments.py --trainable_layers all`
- Row 6: `python additional_experiments.py --model_size "gpt2-medium (355M)"`
- Row 7: `python additional_experiments.py --model_size "gpt2-large (774M)"`
- Row 8: `python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- Row 9: `python additional_experiments.py --model_size "gpt2-xl (1558M)"--trainable_layers all`
- Row 10: `python additional_experiments.py --weights random --trainable_layers all`
- Row 11: `python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- Row 12: `python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- Row 13: `python additional_experiments.py --context_length "model_context_length"`
- Row 14: `python additional_experiments.py --no_padding --batch_size 1`
- Row 15: `python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- Row 16: `python additional_experiments.py --trainable_token_pos "flexible"`
- Row 17: `python additional_experiments.py --disable_causal_mask`
- Row 18: `python additional_experiments.py --ignore_index 50256`
- Row 19: `python additional_experiments.py --average_embeddings`

我有意保持 LLM 和 dataset 较小，所以即使没有 GPU，也可以在普通笔记本（如 MacBook Air M3）上约 15 分钟内完成默认设置的训练。

&nbsp;

### 结果解读（Interpretation）

1. **训练最后一个与第一个 output token position（Row 1 vs. 2）**：训练最后一个 output token position 的表现明显优于训练第一个。这一改进符合预期，原因是 causal self-attention mask。
2. **训练最后一个 Transformer block 与最后一层（Row 1 vs. 3）**：训练整个最后一个 transformer block 的效果也明显好于只训练最后一层。
3. **训练最后一个与最后两个 Transformer blocks（Row 1 vs. 4）**：训练最后两个 transformer blocks 而不是只训练最后一个 block，会带来明显的 3.33% accuracy 提升。
4. **训练最后一个 Transformer block 与所有 layers（Row 1 vs. 5）**：训练所有 layers 相比只训练最后一个 transformer block 有约 2% 的适度提升，但训练时长几乎增加到 3 倍。此外，它的表现不如只训练 12 个 transformer blocks 中最后两个。
5. **使用更大的 pretrained models（Row 1 vs. 6，以及 Row 1 vs. 7 和 8）**：使用大 3 倍的 pretrained model 结果更差。不过，使用大 5 倍的模型会如预期那样改善性能；同样，大 12 倍的模型进一步改善预测性能。（medium model 可能预训练得不够好，或者这个特定 finetuning configuration 对该模型不太合适。）
6. **使用 random weights 与 pretrained weights（Row 1 和 5 vs. 10）**：使用 random weights 的模型结果只比使用 pretrained weights 略差（分别差 3% 和 1.3%）。
7. **使用 LoRA（Low-Rank Adaptation）与训练所有 layers（Row 11 vs. 5，以及 Row 12 vs. 9）**：保持模型冻结并添加 trainable LoRA layers（详见 [Appendix E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)）是训练所有模型参数的可行替代方案，甚至能提升 1 个百分点（row 11 vs. 5）。从使用 LoRA 时 training 与 validation accuracy 的差距约低 1% 可以看出，这可能是因为 overfitting 更少。此外，LoRA 也更 memory-efficient，因为需要更新的参数更少。在训练更大模型时（row 12 vs. 9），也可以看到 LoRA 训练更快（5.79 min 而不是 8.12 min）。
8. **把输入 padding 到完整 context length 与 padding 到最长 training example（Row 1 vs. 13）**：把输入 padding 到模型支持的完整 context length 会显著变差。
9. **Padding 与 no padding（Row 1 vs. 14、15 和 16）**：`--no_padding` 选项会禁用 dataset 中的 padding；由于 inputs 具有可变长度，这要求以 batch size 1 训练模型。这样 test accuracy 更好，但训练时间更长。第 15 行额外启用 8 步 gradient accumulation，以获得与其他实验相同的 batch size，这有助于减少 overfitting，并略微提升 test set accuracy。第 16 行仍然使用 padding，但 token position 根据最后一个 non-padding token 选择。第 16 行在数学上应与使用 gradient accumulation 的第 15 行类似；不过，在 token 数量不相等的情况下，gradient accumulation 会有一些挑战，因此可能存在小差异（[这篇](https://unsloth.ai/blog/gradient)博客文章讨论了这个问题）。
10. **禁用 causal attention mask（Row 1 vs. 17）**：禁用 multi-head attention module 中使用的 causal attention mask。这意味着所有 tokens 都可以 attend 到所有其他 tokens。与带 causal mask 的 GPT model 相比，模型 accuracy 略有提升。
11. **在 loss 和 backpropagation 中忽略 padding indices（Row 1 vs. 18）**：设置 `--ignore_index 50256` 会在 PyTorch 的 `cross_entropy` loss function 中排除 `<|endoftext|>` padding tokens。在这个例子中，它没有影响，因为我们替换了 output layers，使 token IDs 只可能是 0 或 1，用于 binary classification。不过，在第 7 章进行 instruction finetuning models 时，这个设置很有用。
12. **对所有 tokens 的 embeddings 取平均（Row 1 vs. 19）**：设置 `--average_embeddings` 会对所有 tokens 的 embeddings 取平均。如果不使用该选项（默认），只考虑所选 token position（由 `--trainable_token_pos` 指定）上的 output embeddings，例如最后一个 token 的 embeddings。启用 `--average_embeddings` 后，会把所有 tokens 的 embeddings mean-pool 到 `--trainable_token_pos` 选择的位置（默认是最后一个 token）。可以看到，这把性能从 95.00% 提升到 96.33%，运行时间只从 0.28 min 最小幅度增加到 0.32 min，因此实践中值得考虑。
