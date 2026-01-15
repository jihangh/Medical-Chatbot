from app.services.data_loader import load_pdf
from app.services.data_processor import medical_filter_docs

if __name__ == "__main__":
    
    #pdf url and name
    url = "https://staibabussalamsula.ac.id/wp-content/uploads/2024/06/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
    pdfname = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
    
    #load PDF into documents
    pdf_docs = load_pdf(url, pdfname)

    #filter and preprocess documents
    medical_filtered_docs= medical_filter_docs(pdf_docs)