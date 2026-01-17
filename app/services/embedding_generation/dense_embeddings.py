from app.utils.loggers import get_logger
logger = get_logger(__name__)


#generate dense embeddings
def generate_dense_embeddings(lines_batch, dense_model,dim, openai_client): 
    try:
        res = openai_client.embeddings.create(input=lines_batch, model=dense_model, dimensions=dim)
        dense_embeddings = [record.embedding for record in res.data]
        
    except Exception as e:
        logger.error(f"Error generating dense embeddings: {e}")
        raise e
    return dense_embeddings