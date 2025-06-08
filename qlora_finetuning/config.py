from datetime import datetime


def get_default_config():
    now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    user = "arsubhanpuram"
    project = "LLM_FINETUNING"
    run_name = f"{project}-{now}"
    return {
        "seed": 42,
        "base_model": "Qwen/Qwen2.5-3B",
        "dataset_name": "ed-donner/pricer-data",
        "project_name": project,
        "run_name": run_name,
        "run_dir": run_name,
        "hub_model_name": f"{user}/{run_name}",
        "max_seq_length": 2000,
        "epochs": 1,
        "batch_size": 4,
        "grad_acc_steps": 1,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "optimizer": "paged_adamw_32bit",
        "save_steps": 2000,
        "log_steps": 50,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }