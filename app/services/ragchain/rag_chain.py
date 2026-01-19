from app.services.retriever.build_retriever import retrieve_docs
from app.config.config import RAGConfig
from langchain.agents.middleware import dynamic_prompt, ModelRequest
import openai
from openai import OpenAI
from langchain.agents import create_agent
from pinecone.grpc import PineconeGRPC as Pinecone
from pathlib import Path
from app.utils.loggers import get_logger

logger= get_logger(__name__)

#global sys_config
sys_config= RAGConfig.from_yaml("app/config/config.yaml")

@dynamic_prompt
def prompt_with_context(request: ModelRequest ) -> str: #config: RAGConfig
    
    last_query = request.state["messages"][-1].text

    retrieved_docs = retrieve_docs(
        pinecone_vector_client=sys_config.pinecone_vector_client,
        index_name=sys_config.index_name,
        name_space=sys_config.name_space,
        openai_client=sys_config.openai_client,
        dense_model=sys_config.dense_model,
        dim=sys_config.dim,
        query=last_query,
        top_ret_doc=sys_config.top_ret_doc,
        alpha=sys_config.alpha,
    )


    docs_content = "\n\n".join(doc for doc in retrieved_docs)
    

    PROMPT_PATH = Path("prompts/system_prompt.txt")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        sys_prompt = f.read()


    system_message = (
        f"{sys_prompt}"
        f"\n\n{docs_content}"
    )

    return system_message


def rag_assistant(query, prompt_with_context, model, history=None):
    try:
        final_answer = []
        agent = create_agent(model, tools=[], middleware=prompt_with_context)
        for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}}, 
            stream_mode="values",
        ):
            
            msg = step["messages"][-1]

            if hasattr(msg, "content"):
                text = msg.content
            elif isinstance(msg, dict):
                text = msg.get("content")
            else:
                text = str(msg)

            if text:
                final_answer.append(text)
        
       
            
        answer = "".join(final_answer[-1]).strip()
    except Exception as e:
        logger.error('Failed to run the rag chain, error: {}'.format(e))
        raise Exception('RAG chain execution failed: {}'.format(e))
    return answer if answer else "I'm sorry, I couldn't find an answer to your question."