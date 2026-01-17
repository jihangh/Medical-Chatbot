from app.utils.loggers import get_logger

logger = get_logger(__name__)

# Convert the chunk_text into sparse vectors
def generate_sparse_embeddings(pc, lines_batch_chunk):
    try:
        sparse_embeddings = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=[d.page_content for d in lines_batch_chunk],
        parameters={"input_type": "passage", "truncate": "END"})
    except Exception as e:
        logger.error(f"Error generating sparse embeddings: {e}")
        raise e
    return sparse_embeddings