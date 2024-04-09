import os
# import csv
import itertools
from dotenv import load_dotenv
from utils import load_embeddings_from_csv, save_embedding_to_csv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone


def chunks(iterable, batch_size=100):
    # A helper function to break an iterable into chunks of size batch_size.
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


if __name__ == '__main__':
    load_dotenv()

    pinecone_index = os.environ['PINECONE_INDEX_NAME']

    try:
        # Read and create embeddings
        pdf_file = '/Users/joaquinguzman/Desktop/fermax/94726Eb-1 Libro Tecnico MDS Digital V06_10.pdf'
        loader = PyPDFLoader(pdf_file)
        data = loader.load()

        # Create array of Document objects
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(data)

        # Embedding of OpenAI
        embeddings_model = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

        # Create vectors with their text
        embedding_list = []
        for i in range(len(texts)):
            # embedding_list.append(embeddings_model.embed_query(texts[i].page_content))
            embedding = embeddings_model.embed_query(texts[i].page_content)
            document_data = {
                'embedding': embedding,
                'text_content': texts[i].page_content
            }
            embedding_list.append(document_data)

        # Save embeddings to a CSV file
        csv_file_path = '/Users/joaquinguzman/Desktop/fermax/embeddings.csv'
        save_embedding_to_csv(csv_file_path=csv_file_path, embeddings_list=embedding_list)
        '''
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['Document Index', 'Embedding'])
            # Write embeddings
            for ind, embedding in enumerate(embedding_list):
                writer.writerow([ind, embedding])
        '''

        # Load embeddings from CSV
        '''
        csv_path = '/Users/joaquinguzman/Desktop/fermax/embeddings.csv'
        embeddings = load_embeddings_from_csv(csv_file_path=csv_path)
        '''

        # Init Pinecone
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        index = pc.Index(pinecone_index)

        # Upsert data to Pinecone
        for ids_vectors_chunk in chunks(embedding_list, batch_size=100):
            vectors = [{
                'id': str(doc_index),
                'values': embedding,
                'metadata': text_content} for doc_index, embedding, text_content in ids_vectors_chunk]
            index.upsert(vectors=vectors, namespace='libro-tecnico-fermax')

        # Check data type of embeddings
        '''
        for document_index, embedding in embeddings:
            print(f"Document Index: {document_index}, Embedding Type: {type(embedding)}")
        '''

        # Delete a namespace
        '''
        namespace = ''
        index.delete(namespace=namespace, delete_all=True)
        '''

        print(index.describe_index_stats())
    except Exception as e:
        print('The error is: ', e)
