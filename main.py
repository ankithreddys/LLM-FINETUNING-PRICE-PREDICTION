from qlora_finetuning.config import get_default_config
from qlora_finetuning.runner import run_finetuning
import os

if __name__ == "__main__":
    config = get_default_config()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable not set")
    run_finetuning(config, hf_token)