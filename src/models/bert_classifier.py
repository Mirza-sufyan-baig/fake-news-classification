import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class FakeNewsDataset(Dataset):
    
    def __init__(self, texts, labels, tokenizer, max_len = 256):
    
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text, padding = "max_length", truncation = True, 
            max_length = self.max_len, return_tensors ="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        
        return item
    
    
    def compute_metrics(self,pred):
        
        labels = pred.label_ids
        preds = pred.prediction.argmax(-1)
        
        precision , recall, f1, _ = precision_recall_fscore_support(
            labels,preds,average = "binary"
        )
        
        acc = accuracy_score(labels, preds)
        
        return {
            "accuracy" : acc,
            "f1" : f1,
            "precision" : precision,
            "recall" : recall
        }
        
    def load_data(self,file_path):
        df = pd.read_csv(file_path)
        
        df = df.dropna(subset = ["text", "label"])
        
        df["label"] = df["label"].map({
            "fake" : 1,
            "real" : 0
            })    
        
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        
        return texts, labels    
    
    def train_bert(self,file_path):
        
        texts, labels = self.load_data(file_path)
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size = 0.1, random_state = 42
        )
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        train_dataset = FakeNewsDataset(
            train_texts,
            train_labels,
            tokenizer
        )        
        
        val_dataset = FakeNewsDataset(
            val_texts,
            val_labels,
            tokenizer
        )
        
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels = 2
        )
        
        training_args = TrainingArguments(
            output_dir = "model/bert_output",
            learning_rate = 2e-5,
            per_divice_train_batch_size = 8,
            per_device_eval_batch_size  = 8,
            num_train_epochs =3,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            logging_dir = "logs",
            load_best_model_at_end = True,
            metric_for_best_model = "f1"
         )
        
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            compute_metrics = self.compute_metrics
        )
        
        trainer.train()
        trainer.evaluate()
        trainer.save_model("models/bert_fakenews_model")
        
if __name__ == "__main__":
    
    file_path = "data/raw/fake_news_dataset.csv"
    
    classifier = FakeNewsDataset([], [], BertTokenizer.from_pretrained("bert-base-uncased"))
    classifier.train_bert(file_path)