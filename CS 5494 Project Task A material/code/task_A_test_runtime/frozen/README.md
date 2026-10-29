# Task A 正式测试冻结版本

冻结编号：TaskA_baseline_fp16_evaluation_v2_20261027_v1
冻结时间：2026-10-27T15:51:56+08:00（Asia/Shanghai）

本版本只冻结 A。当前阶段已固定设置，尚未读取测试集、尚未生成正式测试结果。

固定 Qwen/Qwen3-1.7B 与 tokenizer 的 commit 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e；FP16、SDPA、无量化、无示例、无 adapter；seed 5494，batch size 1，关闭 thinking，贪心生成，输出上限 256、输入上限 3840、总预算 4096。所有提示词及参数完整保存在 config_frozen.json。project/common/config.json 是相同内容的默认配置。

评估固定 evaluation_v2，QA/RAG 使用 EM、token F1，数学使用数值准确率（绝对容差 1e-6，相对容差 0）。现有验证集结果：QA EM 13.00%，RAG EM 34.6875%，数学 15.33101045%；这些是 907 条验证结果，不是最终测试结果。evidence/ 中保留选择依据及原配置。environment_observed.json 是早期历史记录，当前环境要求以 environment_lock.json 和完整验证 GPU 环境为准。

## 如何使用

1. 将整个冻结包解压到独立目录，在该目录运行 `python verify_freeze.py`。它只校验文件和配置，不需要 GPU。
2. Colab 安装 project/common/requirements_inference.txt 中的固定依赖后，运行 `python verify_freeze.py --check-runtime`，核对 PyTorch 2.11.0+cu130、CUDA 13.0、Tesla T4 及关键依赖。此命令不加载模型权重。
3. 下一步制作正式测试 Colab 入口，必须导入本目录 project/common/ 下的代码并加载 config_frozen.json，遵守 method_contract.json。旧的 907 条验证脚本含验证集专用检查，不能直接当作 1007 条正式测试入口。
4. 正式测试先检查全部输入长度，再用 method=baseline 完成全部预测并保存原始日志；最后才能读取 test_answer.csv，运行本目录的 evaluate.py，以 `--split test --model-config config_frozen.json` 进行评估。

本包不含测试输入、测试参考答案或模型权重。输入/权重将在正式测试准备阶段按固定来源接入。文件发生变化时校验会失败。SHA-256 用于检查副本是否一致，不代替最终运行日志。不要覆盖这份 v1；若必须修改，保留本版和原因，以新编号重新冻结。

当前 common/ 共享开发目录及已交付 B/C 的 ZIP 均不受本次冻结影响。仅将开关改为 true 不能代替完整文件校验。
