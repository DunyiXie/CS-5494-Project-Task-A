# 第四步第四小步：完整验证集 Baseline

长度预检已通过真实 tokenizer 执行和 44 项独立产物核验：907 条输入全部合格，最大 649 tokens；预留 256 个输出 token 后最多 905，当前总预算为 4,096。

本轮对全部 907 条验证题进行真实 GPU 生成和统一评分。使用已通过 10 题实跑的公共 inference.py / evaluation.py，不修改模型权重，不添加示例，不加载 adapter，不使用正式测试集。

## 操作

继续使用原来的 Colab，保持 GPU 开启。取得本地 `project/step4_4_full_validation_baseline.zip`，在底部新建代码单元：

```python
from google.colab import files
from io import BytesIO
from pathlib import PurePosixPath
import zipfile

uploaded_validation = files.upload()
assert len(uploaded_validation) == 1, "请只选择完整验证集 Baseline ZIP"
with zipfile.ZipFile(BytesIO(next(iter(uploaded_validation.values())))) as archive:
    for name in archive.namelist():
        member = PurePosixPath(name)
        assert not member.is_absolute() and ".." not in member.parts
        assert member.parts[0] == "step4_full_validation", "请选择第四小步的完整验证运行包"
    archive.extractall("/content")
```

在另一个代码单元运行：

```python
%run -i /content/step4_full_validation/project/step4_baseline_validation/run_colab_full_validation.py
```

此处 `-i` 允许检查并复用 notebook 中符合要求的 fp16_model/model 和 tokenizer。若对象不存在或不符合配置，则重新加载固定快照。文件名被 Colab 添加 `(1)` 不影响上传内容读取。

本次会实际生成 907 条回答，运行时间会长于 10 题诊断。控制台先核对环境及全部 prompt，然后显示 `[1/907]` 至 `[907/907]`，等待最后的汇总 JSON。

## 运行前检查

- 固定验证 ID 和顺序：QA 300、RAG 320、数学 287。
- 完整输入和原始 config.json 的 SHA256 与已通过的真实长度预检一致。原始配置文件与重新序列化后的 config_snapshot.json 字节哈希可能不同，按各自来源正确核对。
- 使用当前 tokenizer 重新构造全部 907 条 prompt，逐条核对长度和文本 SHA；任何不一致都会在生成前停止。
- 模型及 tokenizer 固定 commit `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`、FP16、SDPA、cuda:0、greedy、关闭 thinking、batch 1、最大生成 256 tokens。
- torch/CUDA 和关键推理依赖与已验证环境一致。实际环境会先保存，若不一致则明确报错；不自动升级或替换依赖。

输入只含 Sample_id,Task,Question,Context。validation_reference.csv 另存，生成期间阻止读取参考文件；参考哈希校验、复制和统一评分均在全部生成结束后执行。

## 完成标志

```text
"status": "STEP4_4_FULL_VALIDATION_BASELINE_PASSED"
"count": 907
"completed_count": 907
"gpu_execution": "completed"
"prompt_parity_verified_count": 907
"reference_reads_during_inference": 0
"bundle_reference_verified_after_generation": true
```

成功标志验证全量覆盖、真实 CUDA 和评分流程，不要求达到某个分数。错误、空白、无法解析或截断的回答按第一小步规则计分，保留在分母中。不会因模型表现较差而跳过题目或把部分预测标成完整实验。

## 结果文件

程序自动下载 `step4_4_baseline_validation_日期时间_UTC.zip`，包含：

| 文件 | 内容 |
| --- | --- |
| runner_summary.json | 完成状态、907 覆盖、环境与 prompt 检查、分类指标、耗时与显存。 |
| backend_environment.json | 实际模型加载/复用、模型与 tokenizer 来源、依赖及有效生成参数；环境检查失败时也保留。 |
| preflight_recheck.json | 本轮生成前全部 907 条 prompt 的长度和哈希重验。 |
| inference/predictions.csv | 907 行预测，仅 Sample_id,Answer 两列。 |
| inference/raw_outputs.jsonl | 原始回答、提取答案、token IDs、EOS/截断标记及逐题耗时。 |
| inference/prompts.jsonl | 实际 prompt 与配置，用于复现。 |
| inference/run_summary.json、config_snapshot.json、environment.json | 公共推理记录。 |
| evaluation/metrics.json | QA/RAG 的 EM、Token F1；数学数值准确率；验证选择分数。 |
| evaluation/per_sample.csv | 逐题得分与提取状态，供错误分析。 |
| reproduce/ | 实际使用的代码、输入、配置、已通过的长度预检，以及生成结束后加入的验证参考答案。 |

生成计时不含模型加载和预热；生成循环墙钟时间包含循环内分词、答案处理及日志保存。显存是 GPU 会话值，可能包含 notebook 里其他保留对象。

Python 异常或手动中断时会尽量保存失败汇总、已完成的原始日志及 ZIP，不为未完成的运行导出最终预测 CSV。不要把 failed 的运行当成完整结果。

下载未自动开始时，可单独运行：

```python
from google.colab import files
files.download(STEP4_4_VALIDATION_RESULT_ZIP)
```

把下载的结果 ZIP 放回本地 project 文件夹，并把最后的汇总 JSON 发到对话中，不必复制全部 907 行进度。下一步据此核验完整结果、分析三类任务的正确/错误案例，并准备 Task A 报告。正式测试的 1,007 条尚未运行，final_test_frozen 仍为 false。

## 本地验证

- prepare_full_validation.py：从固定验证划分导出分开的题目和参考，并验证可接受答案格式。
- verify_full_validation.py：检查已通过的真实长度数据、环境约束、参考文件读取顺序，并用明确标注的模拟后端跑通 907 条生成至评分。模拟结果不用于模型性能结论。
- build_bundle.py：按白名单打包，包含独立的验证输入和参考，不含训练全集、正式测试数据或模型权重。

公共推理和评分源码保持与已验收 10 题 GPU 实验一致；新代码只负责完整验证集的运行前检查、调度及产物验收。
