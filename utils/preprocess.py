import re
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

lemmatizer = WordNetLemmatizer()

MAX_LEN = 60

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'http\S+', '', text)

    text = text.replace("can't", "cannot").replace("won't", "will not")

    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


def preprocess_text(text, tokenizer):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])

    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    return pad