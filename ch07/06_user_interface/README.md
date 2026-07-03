# 构建用户界面与指令微调 GPT 模型交互（Building a User Interface to Interact With the Instruction Finetuned GPT Model）



该 bonus 文件夹包含运行 ChatGPT-like user interface（类似 ChatGPT 的用户界面）的代码，用于与第 7 章的 instruction-finetuned GPT（指令微调 GPT）交互，如下图所示。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-sft.webp?2)



为了实现这个 user interface，我们使用开源的 [Chainlit Python package](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤 1：安装依赖（Install dependencies）

首先，通过下面命令安装 `chainlit` package：

```bash
pip install chainlit
```

（也可以执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤 2：运行 `app` 代码（Run `app` code）

[`app.py`](app.py) 文件包含 UI code。可以打开并查看该文件以了解更多细节。

该文件会加载并使用我们在第 7 章生成的 GPT-2 weights。因此需要先执行 [`../01_main-chapter-code/ch07.ipynb`](../01_main-chapter-code/ch07.ipynb) 文件。

在终端中执行以下命令以启动 UI server：

```bash
chainlit run app.py
```

运行上述命令后，通常会打开一个新的浏览器标签页，你可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端输出，并将本地地址复制到浏览器地址栏中（地址通常是 `http://localhost:8000`）。
