import glob
import os
from doc2vec import read_file, embed_document


def get_file_embeddings():
    # Returns a list of names in list files.
    txt_files = glob.glob('**/*.txt', recursive=True)
    pdf_files = glob.glob('**/*.pdf', recursive=True)
    docx_files = glob.glob('**/*.docx', recursive=True)
    file_names = txt_files + pdf_files + docx_files

    print("Retrieved files:")
    for filename in file_names:
        print(filename)

    vector_list = []

    for file in file_names:
        v = embed_document(read_file(file))
        vector_list.append(v)

    print("Processing files...")
    return file_names, vector_list


def write_folders(named_folders):
    for folder_name, folder in named_folders.items():
        if not len(folder): continue
        directory = folder_name
        for file in folder:
            new_path = os.path.join(os.path.dirname(os.path.abspath(file)),
                                    directory)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            # print(os.path.join(new_path, os.path.basename(os.path.abspath(file))))
            os.rename(file, os.path.join(new_path,
                                         os.path.basename(os.path.abspath(file))))
        print(f"Moved {len(folder)} files to folder named {folder_name}.")
