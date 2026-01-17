from app.services.retriever.hybrid_retriever import HybridRetriever
from app.services.embedding_generation.query_embeddings import generate_query_embeddings


def retrieve_docs(pinecone_vector_client, index_name, name_space,
                    openai_client, dense_model, dim, query,
                             top_ret_doc, alpha):
    
    dense_query_embedding, sparse_query_embedding = generate_query_embeddings(query=query, dense_model=dense_model, 
                                                                           dim=dim, openai_client=openai_client, 
                                                                           pinecone_vector_client=pinecone_vector_client)
        
    
    """Perform hybrid search on Pinecone index using dense and sparse query embeddings."""
    hybrid_retriever = HybridRetriever(pinecone_vector_client=pinecone_vector_client, 
                                       index_name=index_name, 
                                       name_space=name_space)
    
    query_response= hybrid_retriever.contextual_hybrid_search(
                             dense_query_embedding=dense_query_embedding, sparse_query_embedding=sparse_query_embedding,
                             top_ret_doc=top_ret_doc, alpha=alpha)
    return query_response
