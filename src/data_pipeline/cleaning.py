import pandas as pd
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score,classification_report

df = pd.read_csv("data/raw/fake_news_dataset.csv")
print(df.head())

class BasicTextCleaner:
    def __init__(self): 
        self.le = LabelEncoder()
        self.tf = TfidfVectorizer() 
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

cleaner = BasicTextCleaner()
df['cleaned'] = df['text'].apply(cleaner.clean)
#pd.set_option("display.max_colwidth",None)
#print(df[["text", "cleaned"]].head())

class FakeNewsClassifier:
    def __init__(self,df):
        self.df = df
        self.model = LogisticRegression()
        self.vectorizer = TfidfVectorizer(max_features= 5000)
        self.X_train = None
        self.X_test = None 
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self):
        X = self.df["cleaned_text"]
        y = self.df["label"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)

        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))


# --------------------------
# MAIN EXECUTION
# --------------------------

if __name__ == "__main__":
    df = pd.read_csv("data/raw/fake_news_dataset.csv")

    # Clean text
    cleaner = BasicTextCleaner()
    df["cleaned_text"] = df["text"].apply(cleaner.clean)

    # Encode labels
    df["label"] = df["label"].map({"real": 1, "fake": 0})

    # Train model
    classifier = FakeNewsClassifier(df)
    classifier.prepare_data()
    classifier.train()
    classifier.evaluate()
        