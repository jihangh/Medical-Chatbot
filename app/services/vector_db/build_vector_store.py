from app.services.data_ingestion.data_loader import load_pdf
from app.services.data_ingestion.data_processor import medical_filter_docs
from app.services.data_ingestion.data_chunker import chunk_documents
from app.services.vector_db.vector_store import VectorStoreService

def build_medical_vector_store(pinecone_vector_client, openai_client,
                               index_name, name_space, dim,
                               dense_model, batch_size, sleep_time,
                               url, pdfname):
        #load PDF into documents
        pdf_docs = load_pdf(url, pdfname)

        #filter and preprocess documents
        processed_docs= medical_filter_docs(pdf_docs)

        #chunk the documents
        chunks= chunk_documents(processed_docs)
        #initialize VectorStoreService
        vector_store_service = VectorStoreService(
            pinecone_vector_client=pinecone_vector_client,
            open_ai_client=openai_client,
            index_name=index_name,
            dim=dim,
            name_space=name_space,
            dense_model=dense_model,
            batch_size=batch_size,
            sleep_time=sleep_time
        )
        #create Pinecone vector index if not exists
        vector_store_service.create_vector_index()

        # #generate dense and sparse embeddings and upsert them into Pinecone
        vector_store_service.upsert_vectors(all_chunks=chunks[:1])
        return