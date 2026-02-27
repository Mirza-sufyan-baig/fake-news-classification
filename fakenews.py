import pandas as pd
from collections import counter

class FakeNewsEDA():
    def __init__(self,file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("shape", self.df.shape)
        
    def basic_info(self):
        print("\ncolumns")
        print("\nself.df.columns")
        
        print("\ninfo")
        self.df.info
        
        print("\nmissing values")
        print(self.df.isnull().sum())
        
    def label_analysis(self):
        if "label" in self.df.columns:
            print("\nlabel count:")
            print(self.df["label"].value_count())
            
            print("\nlabel percentage")
            print(self.df["label percentage"].value_count(normalize = True))
            
        
