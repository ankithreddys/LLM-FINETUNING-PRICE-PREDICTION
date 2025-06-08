from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

def build_trainer(config, model, tokenizer, dataset):
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
        bias="none"
    )

    trainer_config = SFTConfig(
        output_dir=config["run_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_acc_steps"],
        optim=config["optimizer"],
        save_steps=config["save_steps"],
        logging_steps=config["log_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=0.001,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=config["warmup_ratio"],
        group_by_length=True,
        lr_scheduler_type=config["lr_scheduler"],
        run_name=config["run_name"],
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=config["hub_model_name"],
        hub_private_repo=True
    )

    collator = DataCollatorForCompletionOnlyLM("Price is $", tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=trainer_config,
        data_collator=collator
    )
    return trainer