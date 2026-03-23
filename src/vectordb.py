import os
import chromadb                                                         #type:ignore
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter     #type:ignore
import torch                                                            #type:ignore
from langchain_huggingface import HuggingFaceEmbeddings                 #type:ignore


class VectorDB:
   

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.embedding_model= HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,    #  d_model = 384
            model_kwargs={"device": device},        
        )





        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")


    def chunk_text(self, text: str, title:str , chunk_size: int = 500) :
        

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          
        chunk_overlap=min(chunk_size/10 , 200 ),        
        separators=["\n\n", "\n", "."]         
        
        )
        
        chunks = text_splitter.split_text(text)

        final_chunks = [ 
            {'content' :content,
             'title' : title,
             'id' : f"{title}-chunk_{i}"
            }
            for i,content in enumerate(chunks)
            ]

        return final_chunks


    def add_documents(self, documents: List , chunck_size:int =500) -> None:

       
        print(f"Processing {len(documents)} documents...")
        

        for publication in documents:

            chunked_publication = self.chunk_text(publication['content'] , publication['title'] ,chunck_size )        # text => list of strings for each chunk
            # Chroma has an internal maximum batch size (e.g. 5461), so we need
            # to send data in smaller batches if a single document produces
            # many chunks (large PDFs, etc.).

            max_batch_size = 5000  # stay safely under Chroma's hard limit

            for start in range(0, len(chunked_publication), max_batch_size):
                batch = chunked_publication[start:start + max_batch_size]

                embeddings = self.embedding_model.embed_documents(
                    [chunk['content'] for chunk in batch]
                )

                batch_text = [chunk['content'] for chunk in batch]
                metadatas = [
                    {'title': chunk['title'], 'id': chunk['id']}
                    for chunk in batch
                ]

                next_id = self.collection.count()
                ids = list(range(next_id, next_id + len(batch)))
                ids = [f"document_{id}" for id in ids]

                self.collection.add(
                    embeddings=embeddings,
                    ids=ids,
                    documents=batch_text,
                    metadatas=metadatas,
                )
            
        print("Documents added to vector database")


    def search(self, query: str, n_results: int = 5):

        query_vector = self.embedding_model.embed_query(query)
    
        # Search for similar content
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            include=["documents", "metadatas" , "distances"]
        )
        

        relevant_chunks = []
        for i, doc in enumerate(results["documents"][0]):
            relevant_chunks.append({
                "content": doc,
                "title": results["metadatas"][0][i]["title"] if results["metadatas"][0][i] else "",
                "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                "id": results["metadatas"][0][i]["id"] if results["metadatas"][0][i] else ""
            })
        
        return relevant_chunks



        
