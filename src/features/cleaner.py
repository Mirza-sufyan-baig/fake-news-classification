import re

class BasicTextCleaner:
    def __init__(self):
        pass

    def clean(self, text):
        # 1. Convert to string and lowercase
        text = str(text).lower()
        
        # 2. Remove URLs (Fixed the regex to catch links properly)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        
        # 3. Remove HTML tags
        text = re.sub(r"<.*?>", "", text)
        
        # 4. Remove punctuation and special characters (Keep letters and spaces)
        # This replaces anything that ISN'T a letter or space with nothing
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        
        # 5. Fix extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text