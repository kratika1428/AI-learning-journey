import re

def clean_text(text):
    text = text.lower() #convert to lower case
    text = re.sub(r"http?://\S+|www\.\S+","", text) #remove urls
    text = re.sub(r"\d+","", text) #remove numbers
    text = re.sub(r"[^A-Za-z\s]","", text) #remove punctuations
    text = " ".join(text.split()) #remove extra spaces

    return text