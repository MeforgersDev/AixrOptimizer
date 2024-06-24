from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from torch.utils.data import Dataset

def download_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(f"./models/{model_name}")
    tokenizer.save_pretrained(f"./models/{model_name}")
    print(f"Model {model_name} downloaded and saved.")

def load_model(model_name, device):
    model = AutoModelForSequenceClassification.from_pretrained(f"./models/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"./models/{model_name}")
    model.to(device)
    return model, tokenizer

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        target = self.targets[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return CustomDataset(df, tokenizer, max_len=128)
