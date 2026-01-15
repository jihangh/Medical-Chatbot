import os
import requests
from app.utils.loggers import get_logger
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader



logger = get_logger(__name__)

DATA_DIR = "data"

#download data if not present
def download_data(url, pdfname):
    '''Download PDF from URL'''
    
    #create data directory if not exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
     #download file
    pdf_path = f"{DATA_DIR}/{pdfname}"
    response = requests.get(url)

    # Check if the download was successful
    if response.status_code == 200:
        with open(pdf_path, "wb") as file:
            file.write(response.content)
        logger.info(f"Success! {pdf_path} downloaded.")
    else:
        #raise error if download fails
        logger.error(f"Failed to download. Status code: {response.status_code}")
        raise Exception(f"Failed to download file from {url}")


def load_pdf(url, pdfname) -> List[Document]:
    """Load PDF document"""
    
    pdf_path = f"{DATA_DIR}/{pdfname}"
    #if pdf not present, download it
    if not os.path.exists(pdf_path):
        download_data(url, pdfname)

    logger.info(f"Loading PDF from {pdf_path}")

    try:
        #load pdf using PyMuPDFLoader
        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()

        logger.info(f"Loaded {len(documents)} pages")

    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise e
    return documents