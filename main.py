from app.services.data_loader import download_data, load_pdf

if __name__ == "__main__":
    
    #pdf url and name
    url = "https://staibabussalamsula.ac.id/wp-content/uploads/2024/06/The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
    pdfname = "The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf"
    
    #load PDF
    load_pdf(url, pdfname)