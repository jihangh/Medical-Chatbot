from app.services.data_ingestion.data_loader import load_pdf
from app.services.data_ingestion.data_processor import medical_filter_docs
from app.services.data_ingestion.data_chunker import chunk_documents
from app.services.vector_store import create_vector_index, upsert_vectors
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI

from tqdm import tqdm
from app.utils.loggers import get_logger

logger = get_logger(__name__)

# Configuration

#pdf url and name
url= "https://staibabussalamsula.ac.id/wp-content/uploads/2024/06/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
pdfname = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
#index name of vector store and namespace
index_name= "medical-chatbot-index"
name_space= "medical-namespace"
#dense embedding dimensions
dim=256
#dense embedding model
dense_model = "text-embedding-3-large"
#batch size for upsert
batch_size= 1
#sleep time between upsert batches
sleep_time= 2


def main():
    try:
    
        #initialize OpenAI and Pinecone clients
        load_dotenv()

        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        #set as environment variable
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


        #initialize OpenAI and Pinecone clients
        openai_client = OpenAI()

        pc = Pinecone()

        #load PDF into documents
        pdf_docs = load_pdf(url, pdfname)

        #filter and preprocess documents
        processed_docs= medical_filter_docs(pdf_docs)

        #chunk the documents
        chunks= chunk_documents(processed_docs)
        #create Pinecone vector index if not exists
        create_vector_index(index_name=index_name, 
                            dim=dim, 
                            pinecone_vector_client=pc)

        #generate dense and sparse embeddings and upsert them into Pinecone
        upsert_vectors(pinecone_vector_client=pc, 
                    name_space=name_space, 
                    index_name=index_name, 
                    all_chunks=chunks, 
                    batch_size=batch_size,
                    dense_model=dense_model,
                    dim=dim, 
                    sleep_time=sleep_time,
                    openai_client=openai_client)
    except Exception as e:
        logger.info(f"Error in main execution: {e}")
        raise e

if __name__ == "__main__":
    
    main()
