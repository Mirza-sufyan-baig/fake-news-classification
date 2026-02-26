import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('fake_news_dataset.csv')
print(len(data))
print(data.head())