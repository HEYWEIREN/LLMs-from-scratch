# 版权所有 (c) Sebastian Raschka，Apache License 2.0（参见LICENSE.txt）。
# “从零构建大型语言模型”的来源
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码：https://github.com/rasbt/LLMs-from-scratch

import argparse
import json
import numpy as np
import os

import requests
import tensorflow as tf
import tiktoken
import torch
from tqdm import tqdm

# 从本地文件导入
from previous_chapters import GPTModel


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # 删除批次维度
    return tokenizer.decode(flat.tolist())


def download_and_load_gpt2(model_size, models_dir):
    # 验证模型尺寸
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义路径
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # 加载设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    # 发送 GET 请求以下载文件
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    # 从 headers 中获取文件总大小，如果不存在则默认为 0
    file_size = int(response.headers.get("Content-Length", 0))

    # 检查文件是否存在且大小相同
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size and file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # 定义读取文件的块大小
    block_size = 1024  # 1 KB

    # 使用总文件大小初始化进度条
    progress_bar_description = os.path.basename(url)
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # 以二进制写入方式打开目标文件
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # 更新进度条


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 为每一层初始化空参数字典
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 迭代检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除大小为 1 的维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过 `model/` 前缀

        # 识别变量的目标字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将变量数组分配给最后一个键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # 与之前一样循环生成：计算 logits，并只保留最后一个时间步
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 新功能：使用 top_k 过滤 logits
        if top_k is not None:
            # 只保留 top_k 个最大值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 新功能：应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature

            # 书外补充：提高 MPS 设备结果可复现性的数值稳定技巧
            # 在 softmax 前减去每一行的最大值
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # 应用 softmax 得到概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 按概率分布采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 否则使用贪心解码，选择 logits 最大值对应的索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # 如果遇到序列结束 token 且指定了 eos_id，则提前停止生成
            break

        # 与之前相同：将采样索引附加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def main(gpt_config, input_prompt, model_size, device):

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    gpt = GPTModel(gpt_config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Generate text with a pretrained GPT-2 model.")
    parser.add_argument(
        "--prompt",
        default="Every effort moves you",
        help="Prompt text used to seed the generation."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for running inference, e.g., cpu, cuda, mps, or auto."
    )

    args = parser.parse_args()


    torch.manual_seed(123)

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = args.prompt
    DEVICE = torch.device(args.device)

    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)


    BASE_CONFIG = {
        "vocab_size": 50257,     # 词汇量
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,        # dropout 比率
        "qkv_bias": True         # QKV 偏置
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size, DEVICE)
