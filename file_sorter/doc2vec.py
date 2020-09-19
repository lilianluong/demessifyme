import docx
import fitz
import numpy as np
import pickle as pkl
import time

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation


def load_models():
    t0 = time.time()
    with open("./model/model.pkl", "rb") as f:
        model = pkl.load(f)
    print("Model loaded in", time.time() - t0, "seconds")

    t0 = time.time()
    with open("./model/custom_model.pkl", "rb") as f:
        custom_model = pkl.load(f)
    print("Custom model loaded in", time.time() - t0, "seconds")

    # Load stopwords from nltk
    eng_stopwords = stopwords.words('english')
    print("Loaded list of English stopwords")

    return model, custom_model, eng_stopwords


class DocReader:
    @staticmethod
    def read_docx(filepath):
        doc = docx.Document(filepath)
        text = "\n".join([par.text for par in doc.paragraphs])
        return text

    @staticmethod
    def read_pdf(filepath):
        doc = fitz.open(filepath)
        text = "\n".join([doc[i].getText("text") for i in range(doc.pageCount)])
        doc.close()
        return text

    @staticmethod
    def read_txt(filepath):
        with open(filepath, "r") as f:
            text = f.read()
        return text


def read_file(filepath):
    """
    Get text from file. File must be .txt, .pdf, or .docx

    :param filepath: String, path to text document
    :return: String, text in document
    """
    file_extension = filepath.split(".")[-1].lower()
    read_func = {
        "docx": DocReader.read_docx,
        "pdf": DocReader.read_pdf,
        "txt": DocReader.read_txt
    }[file_extension]
    return read_func(filepath)


def embed_document(text):
    """
    Return a 300-dim embedding of text.

    :param text: String, text to embed
    :return: numpy array with shape=(300,), dtype=np.float
    """
    for punc in punctuation:
        text = text.replace(punc, " ")
    for number in "0123456789":
        text = text.replace(number, " $number ")
    tokens = word_tokenize(text)
    embedding = np.zeros((300,))
    num_tokens = 0
    for token in tokens:
        if token in eng_stopwords:
            continue
        num_tokens += 1
        if token not in model:
            # token doesn't exist in the model, for now skip
            # future approach: generate random vector and save it
            # print(token)
            if token not in custom_model:
                custom_model[token] = np.random.random(size=(300,)) - 0.5
            embedding += custom_model[token]
            continue
        embedding += model[token]
    return embedding / num_tokens


model, custom_model, eng_stopwords = load_models()
