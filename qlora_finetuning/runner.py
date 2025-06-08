import mlflow
from .auth import login_to_huggingface
from .data import load_training_data
from .model import load_model_and_tokenizer
from .trainer import build_trainer


def run_finetuning(config, hf_token):
    login_to_huggingface(hf_token)
    dataset = load_training_data(config["dataset_name"])
    model, tokenizer = load_model_and_tokenizer(config["base_model"])
    trainer = build_trainer(config, model, tokenizer, dataset)

    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(config["project_name"])
    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_params({k: config[k] for k in [
            "learning_rate", "epochs", "batch_size", "optimizer",
            "lora_r", "lora_alpha", "lora_dropout"]})
        trainer.train()
        mlflow.pytorch.log_model(trainer.model, artifact_path="model")
        trainer.model.push_to_hub(config["run_dir"], private=True)