def first_time_setup():
    import gensim
    import pickle as pkl
    import os
    real_path = os.path.dirname(os.path.realpath(__file__))
    model = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(real_path, "model/GoogleNews-vectors-negative300.bin"), binary=True)
    with open(os.path.join(real_path, "model/model.pkl"), "wb") as f:
        pkl.dump(model, f)
    with open(os.path.join(real_path, "model/custom_model.pkl"), "wb") as f:
        pkl.dump({}, f)

    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')
    nltk.download('stopwords')


if __name__ == "__main__":
    first_time_setup()
