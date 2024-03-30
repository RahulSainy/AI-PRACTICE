
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
PINECONE_Index_Name = os.getenv("PINECONE_Index_Name")
PINECONE_Host = os.getenv("PINECONE_Host")


#Extracting Data from the PDF
extracted_data = load_pdf("data/")
#Splitting the text into chunks
text_chunks = text_split(extracted_data)
#Downloading the Embeddings
embeddings = download_hugging_face_embeddings() 


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)
index_name = PINECONE_Index_Name


#Creating Embeddings for Each of The Text Chunks & storing
docsearch=Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)

#If we already have an index we can load it like this
docsearch=Pinecone.from_existing_index(index_name, embeddings)