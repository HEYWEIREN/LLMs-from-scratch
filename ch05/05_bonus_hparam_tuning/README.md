# 优化预训练的超参数（Optimizing Hyperparameters for Pretraining）

[hparam_search.py](hparam_search.py) 脚本基于[附录 D：扩展训练循环](../../appendix-D/01_main-chapter-code/appendix-D.ipynb)中的训练函数，通过网格搜索寻找较优的超参数组合。

>[!NOTE]
该脚本将需要很长时间才能运行。您可能希望减少顶部 `HPARAM_GRID` 字典中探索的超参数配置的数量。
