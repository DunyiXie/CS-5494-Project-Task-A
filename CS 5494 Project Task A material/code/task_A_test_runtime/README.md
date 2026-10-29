# Task A 正式测试：Colab 运行说明

本次只执行 A 的原始模型 baseline：固定 Qwen3-1.7B revision、FP16、无示例、无 adapter，使用冻结的提示词和 evaluation_v2。冻结编号为 `TaskA_baseline_fp16_evaluation_v2_20261027_v1`。本说明和 Notebook 是运行入口；在得到实际 Colab 输出前，不能视为正式测试已完成。

## 准备文件

- 打开新的 Colab Notebook，上传并打开 `notebooks/02_task_A_final_test.ipynb`。
- 将运行时硬件设为 **T4 GPU**。保留此前的验证 Notebook 和结果。
- 第一阶段只上传 `CS5494_TaskA_Final_Test_Colab.zip`。其中包含 1,007 条测试输入及冻结代码，不含测试参考答案或模型权重。
- 将本地 `dataset/test_answer.csv` 留到第二阶段再上传。

Notebook 按实际上传返回的文件名读取 ZIP，因此 Colab 为重名文件添加 `(1)` 时仍可使用。ZIP SHA-256 应与 `bundle_record.json` 一致；正式交付的 Notebook 已写入这个校验值。

## 第一阶段：生成全部预测

按顺序运行 Notebook 前半部分：

1. 上传运行包，核对 ZIP SHA-256 和安全路径，将其解压到 `/content/task_A_test_runtime`。
2. 安装冻结的 Hugging Face 依赖，校验冻结文件及实际环境。运行环境要求 PyTorch `2.11.0+cu130`、CUDA `13.0`、Tesla T4；关键依赖为 Transformers `5.20.2`、Accelerate `1.15.0`、Hugging Face Hub `1.33.0`、Tokenizers `0.23.3`、Safetensors `0.8.0`。
3. 执行：
   ```python
   %run -i /content/task_A_test_runtime/run_colab_task_A_test.py
   ```
   入口检查全部输入长度并使用冻结版本生成。保存逐条原始回答、长度信息、预测、环境、日志和运行摘要。
4. 等待 `TASK_A_TEST_INFERENCE_COMPLETE_PENDING_SCORING`，确认完成 1,007 条预测。下载第一阶段预测 ZIP，作为读取参考答案前的留档。

如果解压目录已经存在且非空，Notebook 会停止，以免混用旧代码。新建一个运行会话后从上传运行包开始。若关键依赖已在旧会话导入，安装后的内存版本可能仍为旧版；重启运行会话后重新执行环境检查。环境检查失败时先恢复记录的环境，不修改冻结配置来绕过检查。

## 第二阶段：计算正式指标

仅在第一阶段成功并留档后，运行 Notebook 后半部分：

1. Notebook 先检查第一阶段完成状态，再显示文件上传控件。
2. 单独选择 `dataset/test_answer.csv`。上传文件名有重名后缀时，代码使用真实返回路径。
3. 执行：
   ```python
   %run -i /content/task_A_test_runtime/score_task_A_test.py --run-dir "$TASK_A_TEST_RUN_DIR" --references "$reference_path"
   ```
   评分入口先检查完整预测和原始日志，再读取参考答案。所有空回答、错误和截断回答保留在分母中。
4. 下载最终结果 ZIP，保存 Notebook，记录其只读共享链接，并截取真实 GPU 环境与成功汇总输出。

正式成绩以实际生成的汇总 JSON 为准。不得根据测试答案或测试分数修改模型、提示词、生成设置、提取规则或评分规则后反复挑选成绩。失败运行需保留日志；同设置重试必须记录原因。

## 交回用于审核的材料

- 第一阶段预测 ZIP；
- 第二阶段最终结果 ZIP；
- 已执行 Notebook 的只读链接；
- 实际环境和正式测试成功汇总截图。

目前交付的 Notebook 没有预填输出和执行次数，不包含任何模拟的 GPU 成功记录。

