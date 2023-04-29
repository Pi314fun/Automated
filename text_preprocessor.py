# text_preprocessor.py
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


def preprocess_classification_and_chatbot(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [ps.stem(token) for token in tokens if token.isalpha()
              and token not in stop_words]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text


def preprocess_text(text: str, task: str) -> str:
    if task == "classification" or task == "chatbot":
        preprocessed_text = preprocess_classification_and_chatbot(text)

    elif task == "sentiment_analysis":
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        preprocessed_text = " ".join(tokens)

    elif task == "entity_extraction":
        doc = nlp(text)
        tokens = [token.text for token in doc]
        preprocessed_text = " ".join(tokens)

    elif task == "product_description":
        doc = nlp(text)
        tokens = [token.text for token in doc if token.is_alpha()]
        preprocessed_text = " ".join(tokens)

    else:
        raise ValueError(
            "Invalid task. Must be one of: classification, sentiment_analysis, entity_extraction, chatbot, product_description")

    return preprocessed_text
