# 第 7 章：微调以遵循指令（Finetuning to Follow Instructions）

该文件夹包含可用于 model evaluation（模型评估）的工具代码。



&nbsp;
## 使用 OpenAI API 评估指令响应（Evaluating Instruction Responses Using the OpenAI API）


- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) notebook 使用 OpenAI 的 GPT-4 来评估 instruction-finetuned models（指令微调模型）生成的 responses（响应）。它使用如下格式的 JSON 文件：

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- The target given in the test set
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- Response by an LLM
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- Response by a 2nd LLM
},
```

&nbsp;
## 使用 Ollama 在本地评估指令响应（Evaluating Instruction Responses Locally Using Ollama）

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) notebook 提供了上面方法的替代方案：通过 Ollama 使用本地下载的 Llama 3 model。
