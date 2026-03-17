from scripts.data_ingestion_validation import run_pipeline as run_task_1
from scripts.eda_analysis import run_pipeline as run_task_2
from scripts.baseline_model import run_pipeline as run_task_3
from scripts.model_improvement import run_pipeline as run_task_4


if __name__ == "__main__":
    run_task_1()
    run_task_2()
    run_task_3()
    run_task_4()
