import datasets
import random
import json
import os.path as osp
import ssl
import urllib.request
import os
def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

def get_dataset(dataset_name: str, max_samples :int, num_fewshots: int = 0):
    
    return_data = []
    fewshot_data = []
    if dataset_name == "gsm8k":
        test_dataset = datasets.load_dataset("openai/gsm8k", "main", split="test[:{}]".format(1000))
        fewshot_dataset = datasets.load_dataset("openai/gsm8k", "main", split="test[{}:]".format(1000))
        for patch in fewshot_dataset:
            text = "Question: " + patch["question"] + "\nAnswer: " + patch["answer"] + "\n"
            fewshot_data.append(text)
        
        few_shot_list = list(range(len(fewshot_data)))
        for patch in test_dataset:
            text = "Question: " + patch["question"] + "\nAnswer: "
            random.shuffle(few_shot_list)
            prefix = ""
            for i in range(num_fewshots):
                prefix += fewshot_data[few_shot_list[i]]
            
            text = prefix + text
            return_data.append(text)
        
        return return_data[:max_samples]

    elif dataset_name == "orca_math":
        test_dataset = datasets.load_dataset("microsoft/orca-math-word-problems-200k", split="train[:1000]")
        fewshot_dataset = datasets.load_dataset("microsoft/orca-math-word-problems-200k", split="train[1000:1500]")
        for patch in fewshot_dataset:
            text = "Question: " + patch["question"] + "\nAnswer: " + patch["answer"] + "\n"
            fewshot_data.append(text)
        
        few_shot_list = list(range(len(fewshot_data)))
        for patch in test_dataset:
            text = "Question: " + patch["question"] + "\nAnswer: "
            random.shuffle(few_shot_list)
            prefix = ""
            for i in range(num_fewshots):
                prefix += fewshot_data[few_shot_list[i]]
            
            text = prefix + text
            return_data.append(text)
        
        return return_data[:max_samples]

    elif dataset_name == "cnn":
        test_dataset = datasets.load_dataset("abisee/cnn_dailymail", "1.0.0", split="test[:2000]")
        fewshot_dataset = datasets.load_dataset("abisee/cnn_dailymail", "1.0.0", split="test[2000:2500]")
        for patch in fewshot_dataset:
           
            text = "###\nArticle: " + patch["article"] + "\n\nSummarize the above article in 1 sentence.\n" + patch["highlights"] + "\n"
            fewshot_data.append(text)

    
        few_shot_list = list(range(len(fewshot_data)))
        for patch in test_dataset:
            text = "###\nArticle: " + patch["article"] + "\n\nSummarize the above article in 1 sentence.\n"
            random.shuffle(few_shot_list)
            prefix = ""
            for i in range(num_fewshots):
                prefix += fewshot_data[few_shot_list[i]]
            
            text = prefix + text
            return_data.append(text)
        
        return return_data[:max_samples]
    
    elif dataset_name == "xsum":
        test_dataset = datasets.load_dataset("EdinburghNLP/xsum", split="test[:2000]")
        fewshot_dataset = datasets.load_dataset("EdinburghNLP/xsum", split="test[2000:2500]")
        for patch in fewshot_dataset:
           
            text = "###\nArticle: " + patch["document"] + "\n\nSummarize the above article in 1 sentence.\n" + patch["summary"] + "\n"
            fewshot_data.append(text)

    
        few_shot_list = list(range(len(fewshot_data)))
        for patch in test_dataset:
            text = "###\nArticle: " + patch["document"] + "\n\nSummarize the above article in 1 sentence.\n"
            random.shuffle(few_shot_list)
            prefix = ""
            for i in range(num_fewshots):
                prefix += fewshot_data[few_shot_list[i]]
            
            text = prefix + text
            
            return_data.append(text)
        
        return return_data[:max_samples]
    
    elif dataset_name == "mtbench":
        test_filepath = os.path.join("mt_bench.jsonl")
        print(f"Loading data from {test_filepath} ...")

        if not os.path.exists(test_filepath):
            download_url(
                "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
                "./"
            )
            os.rename(os.path.join("question.jsonl"), test_filepath)

        list_data = load_jsonl(test_filepath)
        prompts = []
        #base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>{usr_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        for sample in list_data:
            conv = "User: [INST]" + sample["turns"][0] + "[/INST]. \nAssistant: "
            prompts.append(conv)
        return prompts


    elif dataset_name == "python":
        test_dataset = datasets.load_dataset("flytech/python-codes-25k", split="train[:2000]")
        fewshot_dataset = datasets.load_dataset("flytech/python-codes-25k", split="train[2000:2500]")
        for patch in fewshot_dataset:
           
            text = "###\nProblem: " + patch["instruction"] + "\n\nWrite a python program to solve the problem.\n Code: " + patch["output"] + "\n"
            fewshot_data.append(text)

    
        few_shot_list = list(range(len(fewshot_data)))
        for patch in test_dataset:
            text = "###\nProblem: " + patch["instruction"] + "\n\nWrite a python program o solve the problem.\n Code: "
            random.shuffle(few_shot_list)
            prefix = ""
            for i in range(num_fewshots):
                prefix += fewshot_data[few_shot_list[i]]
            
            text = prefix + text
            
            return_data.append(text)
        
        return return_data[:max_samples]
