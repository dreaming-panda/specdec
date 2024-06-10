import datasets
import random
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

