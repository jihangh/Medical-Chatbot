import re
from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.loggers import get_logger

logger= get_logger(__name__)



#remove unwanted prefixes (. or :) from chunks
def clean_chunk_prefix(text: str) -> str:
    return re.sub(r'^[.:]\s*', '', text)


#chunk documents into smaller pieces
def chunk_documents(docs) -> List[Document]:
    """Chunk documents into smaller pieces"""

    try:    
        #recursiveCharacterTextSplitter with medical-specific separators
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,         # Ideal for clinical explanations
        chunk_overlap=100,      # Preserves continuity
        separators=["\n\n",             # Section boundaries (strongest)
                    "\n\n",             # Section boundaries (strongest)
                    "\n",               # Paragraph boundaries
                    ".",                # Sentence boundary
                    ";",
                    ",",                # Sentence boundary 
                    " "
                ]
                )
        
        all_chunks = []
        
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
                src= chunk.metadata.get('source')
                pg= chunk.metadata.get('page')
                temp_doc= Document(metadata={"page": pg, "source": src},
                        page_content= clean_chunk_prefix(chunk.page_content))
                all_chunks.append(temp_doc)

        
    except Exception as e:
        logger.error(f"Error in chunk_documents: {e}")
        raise e
    logger.info(f"{len(all_chunks)} Text chunks obtained")
    return all_chunks