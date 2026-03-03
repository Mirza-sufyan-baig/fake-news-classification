import pandas as pd
import re
df = pd.read_csv("fake_news_dataset.csv")

class Training:
    def __init__(self):
        pass
        
    def clean(self,text):
            text = str(text.lower()) 
            text = text.strip() 
            text = re.sub(r"http\s+ | www\s+", "", text) 
            text = re.sub(r"<.*?>", "", text)
            text = re.sub(r"\s+", " ", text) 
            text = re.sub(r"[^a-zA-Z\s]", "", text) 
            return text 
    def preprocess(self,text):
        self.df['label'] = self.df['label'].map({'real':1, 'fake':0}) 
        print(self.df['label'].value_counts())

cleaner = Training()
df['cleaned'] = df['text'].apply(cleaner.clean)      