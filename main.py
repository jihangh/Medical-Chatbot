from email.mime import message
from functools import partial
from app.services.embedding_generation.query_embeddings import generate_query_embeddings
from app.services.vector_db.build_vector_store import build_medical_vector_store
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from app.services.ragchain.rag_chain import prompt_with_context, rag_assistant
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
import gradio as gr
from app.utils.loggers import get_logger
from app.config.config import RAGConfig
logger = get_logger(__name__)

# Configuration

# #pdf url and name
# url= "https://staibabussalamsula.ac.id/wp-content/uploads/2024/06/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
# pdfname = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
# #index name of vector store and namespace
# index_name= 'hybrid-index'  #"medical-chatbot-index"
# name_space= "hybrid-namespace"  #"medical-namespace"
# #dense embedding dimensions
# dim=1024
# #dense embedding model
# dense_model = "text-embedding-3-large"
# #batch size for upsert
# batch_size= 1
# #sleep time between upsert batches
# sleep_time= 2
# #number of top retrieved documents
# top_ret_doc= 3
# #alpha for hybrid retrieval
# alpha= 0.75



def main():
    try:
        
        #build medical vector store
            
        # build_medical_vector_store(pinecone_vector_client=pc,
        #                            openai_client=openai_client,
        #                            index_name=index_name, 
        #                            name_space=name_space, 
        #                            dim=dim,
        #                            dense_model=dense_model, 
        #                            batch_size=batch_size,
        #                            sleep_time=sleep_time,
        #                            url=url, 
        #                            pdfname=pdfname)
        

        #create RAG agent with hybrid retrieval
        query= "What are the symptoms of diabetes?"
        
        # Load config
        config_filepath= "app/config/config.yaml"
        sys_config = RAGConfig.from_yaml(config_filepath)

        # Create LLM object
        model = sys_config.get_llm()


        # Run query
        query = "What are the symptoms of diabetes?"
        
        # Create Gradio Chat Interface to run the rag chatbot
        def gradio_chat(message, history=None):
            return rag_assistant(message, [prompt_with_context], model=model, history=history)

        
        demo = gr.ChatInterface(
            gradio_chat,
            api_name="medicalchat",
            )

        demo.launch()


    
    except Exception as e:
        logger.info(f"Error in main execution: {e}")
        raise e

if __name__ == "__main__":
    
    main()
