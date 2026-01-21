
from app.services.ragchain.rag_chain import build_prompt_with_context, rag_assistant
from app.services.vector_db.build_vector_store import build_medical_vector_store
from app.config.config import RAGConfig
import sys
import gradio as gr
from app.utils.exceptions import  RagChainError, AppBaseException
from app.utils.loggers import get_logger

#logger
logger = get_logger(__name__)


def main():

    # Create Gradio Chat Interface to run the rag chatbot
    def gradio_chat(message, history=None):
        try:
            return rag_assistant(message, [build_prompt_with_context(sys_config)], model=sys_config.get_llm(), history=history)
        except AppBaseException:
            logger.exception("Domain error in Gradio chat interface")
            return "Sorry, I'm having trouble processing your request right now."
        except RagChainError as e:
            logger.exception("Unexpected error in Gradio chat interface")
            return "Sorry, I'm having trouble processing your request right now."

    try:
        # Load config
        config_filepath= "app/config/config.yaml"
        sys_config = RAGConfig.from_yaml(config_filepath) 
           
        #build medical vector store pf the pdf document           
        #build_medical_vector_store(config=sys_config)
    except AppBaseException as abe:
        logger.exception(f"Application error during main execution: {abe}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error in main execution: {e}")
        sys.exit(1)

    # Create Gradio Chat Interface to run the rag chatbot   
    demo = gr.ChatInterface(
        gradio_chat,
        api_name="medicalchat",
        )
    demo.launch()




        

if __name__ == "__main__":
    
    main()
