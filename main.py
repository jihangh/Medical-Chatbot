from app.services.embedding_generation.query_embeddings import generate_query_embeddings
from app.services.vector_db.build_vector_store import build_medical_vector_store
from app.services.retriever.build_retriever import retrieve_docs
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
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

        #build medical vector store
            
        build_medical_vector_store(pinecone_vector_client=pc,
                                   openai_client=openai_client,
                                   index_name=index_name, 
                                   name_space=name_space, 
                                   dim=dim,
                                   dense_model=dense_model, 
                                   batch_size=batch_size,
                                   sleep_time=sleep_time,
                                   url=url, 
                                   pdfname=pdfname)
        

        retrieve_docs(pinecone_vector_client=pc, index_name=index_name, name_space=name_space,
                    openai_client=openai_client, dense_model=dense_model, dim=dim, query="What are the symptoms of diabetes?",
                             top_ret_doc=20, alpha=0.5)

    
    except Exception as e:
        logger.info(f"Error in main execution: {e}")
        raise e

if __name__ == "__main__":
    
    main()
