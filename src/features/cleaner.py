import re

class BasicTextCleaner:
    def __init__(self):
        pass
    def clean(self,text):
        text = str(text).lower()
        text = re.sub(r"http\s+ | www\s+", "", text)
        text = re.sub(r"<.*?>", "",text)
        text = re.sub(r"[a-zA-Z\s]","", text)
        text = re.sub(r"s\+", " ", text)
        text = text.strip()
        return text