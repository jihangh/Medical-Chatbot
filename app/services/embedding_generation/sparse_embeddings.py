

# Convert the chunk_text into sparse vectors
def generate_sparse_embeddings(lines_batch_chunk):
    sparse_embeddings = pc.inference.embed(
        model="pinecone-sparse-english-v0",
        inputs=[d.page_content for d in lines_batch_chunk],
        parameters={"input_type": "passage", "truncate": "END"})
    return sparse_embeddings