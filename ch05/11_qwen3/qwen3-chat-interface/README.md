# 从零实现 Qwen3，带有聊天界面（Qwen3 From Scratch with Chat Interface）



此补充材料文件夹包含用于运行类似 ChatGPT 的用户界面以与预训练的 Qwen3 模型交互的代码。



![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



为了实现这个用户界面，我们使用开源[Chainlit Python包](https://github.com/Chainlit/chainlit)。

 
## 第 1 步：安装依赖项（Step 1: Install dependencies）

首先，我们通过以下方式安装 [requirements-extra.txt](requirements-extra.txt) 列表中的 `chainlit` 包和依赖项



```bash
pip install -r requirements-extra.txt
```



或者，如果您使用的是 `uv`：



```bash
uv pip install -r requirements-extra.txt
```





 

## 步骤2：运行`app`代码（Step 2: Run `app` code）

该文件夹包含2个文件：

1.[`qwen3-chat-interface.py`](qwen3-chat-interface.py)：该文件在思维模式下加载并使用Qwen3 0.6B模型。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：同上，但配置为记住消息历史记录。

（打开并检查这些文件以了解更多信息。）

从终端运行以下命令之一来启动 UI 服务器：



```bash
chainlit run qwen3-chat-interface.py
```



或者，如果您使用的是 `uv`：



```bash
uv run chainlit run qwen3-chat-interface.py
```



运行上述命令之一应该会打开一个新的浏览器选项卡，您可以在其中与模型进行交互。如果浏览器选项卡没有自动打开，请检查终端命令并将本地地址复制到浏览器地址栏中（通常该地址为 `http://localhost:8000`）。