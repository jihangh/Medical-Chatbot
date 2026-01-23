import sys
import uuid
import gradio as gr
from app.services.ragchain.rag_chain import rag_assistant, build_prompt_with_context
from app.services.vector_db.build_vector_store import build_medical_vector_store
from app.config.config import RAGConfig
from app.utils.loggers import get_logger
from app.utils.exceptions import AppBaseException


logger = get_logger(__name__)

session_id = str(uuid.uuid4()) 

def main():

    def chat_fn(message, history=None):
        '''
        Gradio chat function to handle user messages and return responses from the RAG assistant.
        '''
        try:
            # Call your RAG agent; memory is handled internally via session_id
            answer = rag_assistant(
                message,
                [build_prompt_with_context(sys_config)],
                model=sys_config.get_llm(),
                session_id=session_id
            )
            return answer  # ChatInterface expects a string or ChatMessage object

        except AppBaseException:
            logger.exception("Domain error in ChatInterface")
            return "Sorry, I'm having trouble processing your request right now."

        except Exception:
            logger.exception("Unexpected error in ChatInterface")
            return "Sorry, I'm having trouble processing your request right now."

    
    
    # Load configuration
    try:
        config_filepath = "app/config/config.yaml"
        sys_config = RAGConfig.from_yaml(config_filepath)

        # Build your vector store (can be skipped if already built)
        #build_medical_vector_store(config=sys_config)

    except AppBaseException as abe:
        logger.exception(f"Application error during main execution: {abe}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error in main execution: {e}")
        sys.exit(1)


    # Gradio ChatInterface automatically handles the display of user/bot messages

    demo = gr.ChatInterface(
        fn=chat_fn,  # each user gets a unique session_id via closure
        title="Medical Chatbot with RAG",
        description="Ask your medical questions"
    )

    demo.launch(theme=gr.themes.Soft())

if __name__ == "__main__":
    main()


