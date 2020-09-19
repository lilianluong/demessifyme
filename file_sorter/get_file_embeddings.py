import glob
from doc2vec import read_file, embed_document

def get_file_embeddings():

    file_names = []

    # Returns a list of names in list files.
    txt_files = glob.glob('**/*.txt', recursive=True)
    pdf_files = glob.glob('**/*.pdf', recursive=True)
    docx_files = glob.glob('**/*.docx', recursive=True)
    for file in txt_files:
        file_names.append(file)
    for file in pdf_files:
        file_names.append(file)
    for file in docx_files:
        file_names.append(file)

    print(file_names)

    vector_list = []

    for file in file_names:
        v = embed_document(read_file(file))
        vector_list.append(v)

    return file_names, vector_list