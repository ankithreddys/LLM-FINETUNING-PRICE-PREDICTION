from datasets import load_dataset


def load_training_data(dataset_name):
    dataset = load_dataset(dataset_name)
    print("====  DATA LOADED  ====")
    return dataset["train"]