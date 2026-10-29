CS5494 TASK A RUN INSTRUCTIONS

1. SETUP

Open each notebook in Google Colab or Colab Enterprise and connect it to
a separate fresh NVIDIA Tesla T4 GPU runtime. Run the specified cells
individually, rather than using Run all. Internet access is required for
dependency installation and model download.

Prepare these files:
- CS_5494_Task_A_Baseline.ipynb
- step4_4_full_validation_baseline_evaluation_v2.zip
- 02_task_A_final_test.ipynb
- CS5494_TaskA_Final_Test_Colab.zip
- dataset/test_answer.csv

The runtime ZIPs contain the required inputs and code. Upload each ZIP
without extracting it locally.

2. VALIDATION BASELINE

Open CS_5494_Task_A_Baseline.ipynb. Run only the cells identified below
to reproduce the complete validation baseline.

Step 1: Run the dependency-installation cell containing
torch==2.11.0+cu130 and transformers==5.20.2. Wait for installation to
finish, then restart the Python session from the runtime menu.

Step 2: After reconnecting, run the environment-check cell containing
assert torch.__version__ == "2.11.0+cu130". Continue when all assertions pass.

Step 3: Run the cell containing uploaded_validation = files.upload().
Select only step4_4_full_validation_baseline_evaluation_v2.zip.

Step 4: Run the full-validation cell:

%run -i /content/step4_full_validation/project/step4_baseline_validation/run_colab_full_validation.py

Wait for STEP4_4_FULL_VALIDATION_BASELINE_PASSED and completed_count=907.
Download the generated step4_4_baseline_validation_<timestamp>_UTC.zip.
If the automatic download does not start, run:

from google.colab import files
files.download(STEP4_4_VALIDATION_RESULT_ZIP)

3. FINAL TEST

Open 02_task_A_final_test.ipynb in a separate fresh T4 runtime.

Step 1: Run the first code cell, containing EXPECTED_ZIP_SHA256.
Upload only CS5494_TaskA_Final_Test_Colab.zip. Keep test_answer.csv locally
until Step 6.

Step 2: Run the dependency-installation cell containing torch_packages.
After it finishes, manually restart the Python session. Retain the runtime
files; do not delete or factory-reset the runtime.

Step 3: After reconnecting, continue directly at the environment-check cell
containing marker_path = Path("/content/task_A_environment_install.json").
Do not rerun the installer. Continue when FROZEN_ENVIRONMENT_READY appears.

Step 4: Run the prediction-generation cell:

%run -i /content/task_A_test_runtime/run_colab_task_A_test.py

Step 5: Run the completion-check cell containing
run_dir = Path(TASK_A_TEST_RUN_DIR). Confirm
TASK_A_TEST_INFERENCE_COMPLETE_PENDING_SCORING and completed_count=1007.
Save task_A_test_<run_timestamp>_UTC_predictions.zip before proceeding.
If the automatic download does not start, run:

from google.colab import files
files.download(TASK_A_TEST_PREDICTION_ZIP)

Step 6: Stay in the same session. Run the cell containing
reference_upload = files.upload() and select only dataset/test_answer.csv.
Then run the scoring cell:

%run -i /content/task_A_test_runtime/score_task_A_test.py --run-dir "$TASK_A_TEST_RUN_DIR" --references "$reference_path"

Step 7: Confirm TASK_A_FINAL_TEST_PASSED and completed_count=1007.
Download task_A_test_<run_timestamp>_UTC_scored_<scoring_timestamp>_UTC.zip.
If the automatic download does not start, run:

from google.colab import files
files.download(TASK_A_TEST_RESULT_ZIP)

4. OUTPUT FILES

Extract the downloaded result ZIPs to access:
- inference/predictions.csv: predicted answers.
- inference/raw_outputs.jsonl: raw model outputs.
- evaluation/metrics.json: validation scores.
- final_summary.json: final-test scores and completion status.
- scoring_<timestamp>_UTC/evaluation/per_sample.csv: final-test per-example scores.
