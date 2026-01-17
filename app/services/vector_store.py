
from tqdm import tqdm
import time
from app.services.embedding_generation.dense_embeddings import generate_dense_embeddings
from app.services.embedding_generation.sparse_embeddings import generate_sparse_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from app.utils.loggers import get_logger

logger = get_logger(__name__)



def create_vector_index(index_name, dim,pinecone_vector_client):   
    '''Create Pinecone vector index if not exists,
      with dense vector type and dotproduct metric for hybrid search'''
    try:
        if not pinecone_vector_client.has_index(index_name):
            pinecone_vector_client.create_index(
                name=index_name,
                vector_type="dense",
                dimension=dim,
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
    except Exception as e:
        logger.error(f"Error creating vector index: {e}")
        raise e
 

def upsert_vectors(pinecone_vector_client, name_space,index_name, all_chunks, batch_size,
                    dense_model,dim, sleep_time,openai_client):
    '''Upsert vectors into Pinecone index in batches'''
    
    index = pinecone_vector_client.Index(index_name)
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        # set end position of batch
        i_end = min(i + batch_size, len(all_chunks))
        # get batch of lines and IDs
        lines_batch_chunk = all_chunks[i: i_end]
        lines_batch = [d.page_content for d in lines_batch_chunk]
        ids_batch = [str(n) for n in range(i, i_end)]
        
        # create dense embeddings
        dense_embeddings = generate_dense_embeddings(lines_batch, dense_model,dim, openai_client)
        # Convert the chunk_text into sparse vectors
        sparse_embeddings = generate_sparse_embeddings(pinecone_vector_client, lines_batch_chunk)
        
        # prep metadata and upsert batch
        meta = [line.metadata for line in lines_batch_chunk]
        
        # upsert to Pinecone
        # Each record contains an ID, a dense vector, a sparse vector, and the original text as metadata
        records_embed = []
        for d, de, se, m, t in zip(ids_batch, dense_embeddings, sparse_embeddings, meta,lines_batch):
            records_embed.append({
                "id": str(d),
                "values": de,
                "sparse_values": {
                    "indices": se["sparse_indices"],
                    "values": se["sparse_values"]
                },
                "metadata": {
                    **m,
                    "text": t
                }
            })
            

        # Upsert the records
        # The `chunk_text` fields are converted to dense and sparse vectors
        try:
            index.upsert(vectors= records_embed, namespace= name_space)
        except Exception as e:
            logger.error(f"Error upserting vectors of batch {i_end}: {e}")
            raise e
        logger.info(f"Uploaded vectors of batch {i_end}")
        time.sleep(sleep_time)  # Respect rate limits