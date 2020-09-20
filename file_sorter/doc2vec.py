import docx
import fitz
import numpy as np
import os
import pickle as pkl
import time

from collections import Counter
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from string import punctuation


def load_models():
    t0 = time.time()
    real_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(real_path, "model/model.pkl"), "rb") as f:
        model = pkl.load(f)
    print("Model loaded in", time.time() - t0, "seconds")

    # Load stopwords from nltk
    eng_stopwords = stopwords.words('english')
    print("Loaded list of English stopwords")

    return model, {}, eng_stopwords


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
    text = read_func(filepath)
    text += (" " + (".".join(filepath.split(".")[:-1])).split("/")[-1]) * 15  # weigh file name
    return text


def tokenize(text):
    """
    Split text into tokens, removing punctuation and replacing numbers with $number tokens.

    :param text: text to tokenize
    :return: list of tokens
    """
    for punc in punctuation:
        text = text.replace(punc, " ")
    # for number in "0123456789":
    #     text = text.replace(number, " number ")
    # tokens = word_tokenize(text)
    tokens = [("$number" if s.isnumeric() else s) for s in text.lower().split()]
    return tokens


def embed_document(text):
    """
    Return a 300-dim embedding of text.

    :param text: String, text to embed
    :return: numpy array with shape=(300,), dtype=np.float
    """
    tokens = tokenize(text)
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


def get_folder_name(folders):
    """
    Given a list of folders of files, choose a token to name the folder.

    :param folders: list of lists of filenames
    :return: String, name of folder
    """
    num_misc = 0
    names = {"misc": folders[0]}
    for folder in folders[1:]:
        if len(folder) == 1:
            names["misc"].extend(folder)
            continue
        tokens = tokenize("\n".join([read_file(filepath) for filepath in folder]))
        tokens = [t for t in tokens if t not in eng_stopwords]
        found = False
        for (key, freq) in Counter(tokens).most_common():
            if freq < 10:
                break
            if key != "$number" and len(key) > 2:
                names[key] = folder
                found = True
                break
        if not found:
            names["untitled" + str(num_misc + 1)] = folder
            num_misc += 1

    return names


model, custom_model, eng_stopwords = load_models()
