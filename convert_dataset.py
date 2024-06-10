import datasets

def get_dataset(dataset_name: str, max_samples :int):
    
    if dataset_name == "gsm8k":
        dataset = datasets.load_dataset("openai/gsm8k", "main", split="test[:{}]".format(max_samples))
    
    elif dataset_name == "cnn":
        dataset = datasets.load_dataset("abisee/cnn_dailymail", "1.0.0", split="test[:{}]".format(max_samples))
    
    
    return dataset

