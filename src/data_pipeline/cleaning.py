import re 
import nltk
from nltk import stopwords
from nltk import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

class TextCleaner:
    def __init__(self,file_path,text_column):
        self.file_path = file_path
        self.df = None
        self.text_column = text_column
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
    def CleanText(self,text):
        text = str(text)
        text = text.lower()
        return text
        text = re.sub(r"http\s+ | www\s+ | https\s+")
        text = re.sub(r"<\s+|>\s+")
        text = re.sub(r"[^a-zA-Z]", '',text)
        word = text.split()
        filterd_words = []
        if word not in self.stop_words:
            lemma = self.lemmatizer.lemmatize(word)
            filtered_words.append(lemma)
        text = " ".join(filterd_words)
        return text
                
        