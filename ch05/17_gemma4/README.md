# Gemma 4

该目录包含一个仅处理文本的独立 Gemma 4 Notebook。它基于 Gemma 3 参考 Notebook 构建，并适配了稠密的 `google/gemma-4-E2B` 和 `google/gemma-4-E4B` checkpoint。

- [standalone-gemma4.ipynb](./standalone-gemma4.ipynb) 在纯 PyTorch 中实现共享 Gemma 4 密集架构，并通过 `CHOOSE_MODEL` 在 E2B 和 E4B 参考配置之间切换。
