import pandas as pd
import numpy as np
from src.features.cleaner import BasicTextCleaner
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
class Training:
    

    def __init__(self):
        self.X = None
        self.y = None
        self.skf = None
        pass
    def load_and_prepare_data(self,file_path):
        df = pd.read_csv(file_path)
        cleaner = BasicTextCleaner()
        df["cleaned_text"] = df["text"].apply(cleaner.clean)
        df["label"] = df["label"].map({'real' : 1, 'fake' : 0})
        self.X = df["cleaned_text"]
        self.y = df["label"]
    
        return self.X, self.y
    def run_evaluation(self,models):
        self.skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
        results = []
        for model_name, model_object in models.items():
            fold_f1_score = []
            print(f"Evaluating {model_name}...")
            
            for train_index ,test_index in self.skf.split(self.X,self.y):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
                
                tfidf = TfidfVectorizer(max_features = 5000)

if __name__ == "__main__":
    file_path = "data/raw/fake_news_dataset.csv"
    classifier = Training()
    X, y = classifier.load_and_prepare_data(file_path)
    models = {
        "LogisticRegression": LogisticRegression(),
        "Linear svm": LinearSVC(),
        "naive bayes": MultinomialNB()
    }
    
    print("data loaded successfully")
    print("Number of samples", len(X))
    print("class distribution:")
    print(y.value_counts())    