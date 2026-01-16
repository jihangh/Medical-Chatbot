
#generate dense embeddings
def generate_dense_embeddings(lines_batch, DENSE_MODEL,DIMENSIONS, CLIENT):
    res = CLIENT.embeddings.create(input=lines_batch, model=DENSE_MODEL, dimensions=DIMENSIONS)
    dense_embeddings = [record.embedding for record in res.data]
    return dense_embeddings