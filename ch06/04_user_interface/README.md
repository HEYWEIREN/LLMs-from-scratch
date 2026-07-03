# 构建与 GPT-based Spam Classifier 交互的用户界面（User Interface）



这个 bonus 文件夹包含运行 ChatGPT-like user interface 的代码，用于与第 6 章中 finetuned GPT-based spam classifier 交互，如下图所示。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-spam.webp)



为了实现这个 user interface，我们使用开源 [Chainlit Python package](https://github.com/Chainlit/chainlit)。

&nbsp;
## 第 1 步：安装依赖（Install dependencies）

首先，通过下面命令安装 `chainlit` 包：

```bash
pip install chainlit
```

（也可以执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 第 2 步：运行 `app` 代码（Run `app` code）

[`app.py`](app.py) 文件包含基于 Chainlit 的 UI code。打开并查看这些文件，可以了解更多细节。

该文件会加载并使用我们在第 6 章生成的 GPT-2 classifier weights。这要求你先执行 [`../01_main-chapter-code/ch06.ipynb`](../01_main-chapter-code/ch06.ipynb) 文件。

从终端执行下面命令启动 UI server：

```bash
chainlit run app.py
```

运行上面的命令后，应会打开一个新的浏览器标签页，可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端命令并把本地地址复制到浏览器地址栏中（通常地址是 `http://localhost:8000`）。
