# 第 7 章：微调以遵循指令（Finetuning to Follow Instructions）

该文件夹包含用于准备 instruction dataset（指令数据集）的工具代码。

通过下面命令安装额外依赖：

```bash
pip install -r requirements-extra.txt
```





### 查找近似重复项（Finding Near Duplicates）

`find-near-duplicates.py` 函数可用于在 instruction dataset 中识别 duplicates（重复项）和 near-duplicates（近似重复项）。例如：



```bash
python find-near-duplicates.py --json_file instruction-examples.json
```

```
scikit-learn version: 1.3.1


==================================================
Searching 'instruction' for duplicates ...
==================================================
Duplicate pair found with similarity 0.94:
1. Edit the following sentence to make it more formal.
2. Edit the sentence to make it more formal.

Duplicate pair found with similarity 1.00:
1. Name a dwarf planet in our solar system.
2. Name a dwarf planet in our solar system.

Duplicate pair found with similarity 0.91:
1. Change the sentences from active voice to passive voice.
2. Change the sentence from passive to active voice.



==================================================
Searching 'input' for duplicates ...
==================================================
No duplicates found


==================================================
Searching 'output' for duplicates ...
==================================================
Duplicate pair found with similarity 1.00:
1. One dwarf planet in our solar system is Pluto.
2. One dwarf planet in our solar system is Pluto.


```

&nbsp;
可以通过取值在 0 到 1 之间的 `--threshold` 设置来降低或提高敏感度。
默认 threshold（阈值）为 0.9。



&nbsp;
 ## 创建被动语态条目（Creating Passive Voice Entries）

 - [create-passive-voice-entries.ipynb](create-passive-voice-entries.ipynb) notebook 使用 OpenAI 的 GPT-4 为 instruction dataset 创建 "passive voice"（被动语态）条目，如下例所示。

 ```python
 {  
    'instruction': 'Identify the verb in the following sentence',
    'input': 'The cat sleeps on the couch.',
    'output': 'The verb in the sentence is "sleeps."',
    'output_2': 'The sentence is "sleeps."'   #  <---- Newly created entry
 }  
 ```
