import pandas as pd
from collections import Counter


class FakeNewsEDA:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("shape:", self.df.shape)

    def basic_info(self):
        print("\ncolumns:")
        print(self.df.columns)

        print("\ninfo:")
        self.df.info()

        print("\nmissing values:")
        print(self.df.isnull().sum())

    def label_analysis(self):
        if "label" in self.df.columns:
            print("\nlabel counts:")
            print(self.df["label"].value_counts())

            print("\nlabel percentage:")
            print(self.df["label"].value_counts(normalize=True))

    def text_length_analysis(self):
        if "text" in self.df.columns:
            self.df["text_length"] = self.df["text"].astype(str).apply(
                lambda x: len(x.split())
            )

            print("\ntext length stats:")
            print(self.df["text_length"].describe())

            print("avg length:", self.df["text_length"].mean())

    def duplicate_check(self):
        print("\nduplicate rows:", self.df.duplicated().sum())

        if "text" in self.df.columns:
            print("duplicate texts:", self.df.duplicated(subset=["text"]).sum())

    def common_words(self):
        if "text" in self.df.columns:
            words = " ".join(self.df["text"].astype(str)).split()
            common_words = Counter(words).most_common(20)

            print("\ncommon words:")
            for w, c in common_words:
                print(w, c)

    def combine_title_text(self):
        if "title" in self.df.columns:
            self.df["combined_text"] = (
                self.df["title"].astype(str) + " " + self.df["text"].astype(str)
            )

            print("\ncombined sample:")
            print(self.df["combined_text"].iloc[0])



eda = FakeNewsEDA("data/raw/fake_news_dataset.csv")

eda.load_data()
eda.basic_info()
eda.label_analysis()
eda.text_length_analysis()
eda.duplicate_check()
eda.common_words()

print(eda.df["label"].unique())
        
