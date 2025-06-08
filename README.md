# QLoRA Fine-Tuning for Price Prediction

This project provides a modular, MLflow-integrated pipeline for fine-tuning the Qwen-2.5-3B model using QLoRA on a structured dataset designed for product price prediction.

---

## Project Structure

```
├── main.py                        # Entry point to run the training pipeline
├── qlora_finetune/
│   ├── __init__.py
│   ├── config.py                 # Returns training configuration
│   ├── auth.py                   # Handles Hugging Face authentication
│   ├── data.py                   # Loads and preprocesses dataset
│   ├── model.py                  # Loads tokenizer and 4-bit quantized model
│   ├── trainer.py                # Prepares LoRA config and SFTTrainer
│   └── runner.py                 # Full training orchestration
```

---

## Features

* Modular Python components
* QLoRA (4-bit) fine-tuning support
* Hugging Face Hub push
* MLflow logging and tracking

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/ankithreddys/LLM-FINETUNING-PRICE-PREDICTION.git
cd LLM-FINETUNING-PRICE-PREDICTION
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/macOS
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

```bash
# Required for pushing to Hugging Face Hub
set HF_TOKEN=your_huggingface_token
# Optional: Set in shell or .env file
```

---

## Run Training

```bash
python main.py
```

MLflow will log:

* Parameters
* Model artifacts
* Training metrics (loss)
* A PyTorch model you can later serve or evaluate

The model will also be pushed to your Hugging Face Hub as a private model.

---

## 🧠 Core Components

### `config.py`

Defines hyperparameters and model paths.

### `auth.py`

Logs into the Hugging Face Hub using the `HF_TOKEN` env variable.

### `data.py`

Loads the dataset via `datasets.load_dataset()`.

### `model.py`

Loads a 4-bit quantized Qwen-2.5-3B model using `BitsAndBytesConfig`.

### `trainer.py`

Sets up the LoRA config and Hugging Face `SFTTrainer` for supervised fine-tuning.

### `runner.py`

Combines all pieces: login, data load, model prep, training, MLflow logging, and hub push.

---

## Model and Dataset

* **Base Model**: `Qwen/Qwen2.5-3B`
* **Dataset**: Custom dataset hosted on HF Hub (e.g. `ed-donner/pricer-data`)

---

## MLflow Logging

Logs the following to MLflow:

* Learning rate
* Epochs
* LoRA settings (r, alpha, dropout)
* Optimizer
* Model artifact

To start the MLflow UI:

```bash
mlflow ui
```

Open your browser to: [http://localhost:5000](http://localhost:5000)

---

## Notes

* Make sure you have sufficient GPU RAM (\~24GB or more recommended).
* If there are any runtime disconnects, you can resume by reloading from the last pushed hub checkpoint.

---

## Author

**Ankith Reddy Subhanpuram**

* Master's in AI Systems, University of Florida
