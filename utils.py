import csv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()


# Read all files in a given directory
def read_doc(directory):
    documents = []
    files = os.listdir(directory)
    all_files = len(files)
    print(f'There are {all_files} files to scan')

    try:
        for iteration, file in enumerate(files):
            if file is not None:
                # pdf_file = os.path.join(os.path.expanduser('~'), 'Desktop/fermax/01-test') + file
                pdf_file = os.path.join(directory, file)
                loader = PyPDFLoader(pdf_file)
                print(f'File {file} done, {iteration}')
                documents.extend(loader.load())
    except Exception as e:
        print(e)
    return documents


# Returns a list with the text in chunks
def split_text(docs_to_split):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    texts = text_splitter.split_documents(docs_to_split)
    return texts


# Load embeddings from CSV file
def load_embeddings_from_csv(csv_file_path):
    embeddings = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            document_index = int(row[0])
            embedding_str = row[1]
            embedding_list = eval(embedding_str)  # Convert string representation to list
            embeddings.append((document_index, embedding_list))
    return embeddings


def save_embedding_to_csv(csv_file_path, embeddings_list):
    fieldnames = ['Document Index', 'Embedding', 'Text Content']
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, document_data in enumerate(embeddings_list):
            writer.writerow({
                'Document Index': index,
                'Embedding': document_data['embedding'],
                'Text Content': document_data['text_content']
            })
