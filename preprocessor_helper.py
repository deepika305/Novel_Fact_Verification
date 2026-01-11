from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import PromptTemplate
from helper import Model
from time import sleep
# from dotenv import load_dotenv

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_cache(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        return json.load(f)
def save_cache(data, CACHE_FILE):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)
def get_data(CACHE_FILE):
    if os.path.exists(CACHE_FILE):
        print("Loading from cache...")
        return load_cache(CACHE_FILE)
    else:
        print("Cache not found. Generating data...")
        return None

class PrecomputedEmbeddings(Embeddings):
    def __init__(self, doc_embeddings):
        self.doc_embeddings = doc_embeddings

    def embed_documents(self, texts):
        # One embedding per document
        return self.doc_embeddings[:len(texts)]

    def embed_query(self, text):
        raise NotImplementedError("Query embedding is provided directly")

class NovelPreprocessor():
    # constructor
    def __init__(self, novel_path, novel_name="unknown"):
        self.novel_path = novel_path
        self.novel_name = novel_name
        self.book_chunks = []
        self.model = Model()

    # method to break novel to chunks
    def split_book_into_chunks(self, chunk_size: int = 10000, chunk_overlap: int = 200):
        if not os.path.exists(self.novel_path):
            print(f"Error: File not found at {self.novel_path}")
            return []

        try:
            with open(self.novel_path, 'r', encoding='utf-8') as f:
                book_content = f.read()
        except Exception as e:
            print(f"Error reading file {self.novel_path}: {e}")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = text_splitter.create_documents([book_content])
        print(f"Book split into {len(chunks)} chunks.")
        # self.book_chunks = chunks
        # add page content in book_chunks
        self.book_chunks = [chunk.page_content for chunk in chunks]
        with open(f"cache/cache_chunks_{self.novel_name}.json", "w") as f:
            json.dump(self.book_chunks, f)
        return chunks
    
    def process_chunk(self, chunk_text: str, chunk_id: int):
        facts_cache = f"cache/facts_{self.novel_name}_{chunk_id}.json"
        
        # 1. Fact Extraction
        fact_sentences = []
        if os.path.exists(facts_cache):
            # print(f"Loading facts for chunk {chunk_id} from cache...")
            fact_sentences = load_cache(facts_cache)
        else:
            try:
                llm = self.model.get_llm()
                prompt_template = PromptTemplate(
                    input_variables=["sentence"],
                    template="""
                            Extract distinct, atomic facts from the following sentence.
                            List each fact on a new line as a sentence. Do not include introductory phrases or numbers.
                            If no facts can be extracted, return an empty string.
                            Sentence: {sentence}
                            """
                )
                
                # Manual invocation of llm to debug/control better or use the chain
                chain = prompt_template | llm | StrOutputParser()
                extracted_text = chain.invoke({"sentence": chunk_text}).strip()
                
                # Avoid rate limits if necessary
                # time.sleep(1) 
                
                if extracted_text:
                    fact_sentences = [fact.strip() for fact in extracted_text.split('\n') if fact.strip()]
                    save_cache(fact_sentences, facts_cache)
                else:
                    # Save empty to avoid re-running failing chunks? 
                    # Better to not save empty so it retries, unless truly empty content.
                    pass 
                    
            except Exception as e:
                print(f"Error extracting facts for chunk {chunk_id}: {e}")
                return [], []

        if not fact_sentences:
            # print(f"No facts found for chunk {chunk_id}.")
            return [], []

        # 2. Embeddings
        embeddings_cache = f"cache/embeddings_{self.novel_name}_{chunk_id}.json"
        if os.path.exists(embeddings_cache):
            embedding_vectors = load_cache(embeddings_cache)
        else:
            try:
                embeddings_model = self.model.get_embedding_model()
                embedding_vectors = embeddings_model.embed_documents(fact_sentences)
                save_cache(embedding_vectors, embeddings_cache)
            except Exception as e:
                print(f"Error generating embeddings for chunk {chunk_id}: {e}")
                return [], []

        return fact_sentences, embedding_vectors

    def forward(self):
        print("Starting incremental processing...")
        if not os.path.exists("cache"):
            os.makedirs("cache")
            
        chunks = self.split_book_into_chunks(chunk_size=10000)
        if not chunks:
            print("No chunks to process.")
            return

        tracker_file = f"cache/processed_chunks_{self.novel_name}.json"
        processed_ids = set()
        if os.path.exists(tracker_file):
            try:
                processed_ids = set(load_cache(tracker_file))
                print(f"Loaded progress: {len(processed_ids)} chunks already processed.")
            except:
                pass

        index_path = f"{self.novel_name}_faiss_index"
        vectorstore = None
        
        # Try loading existing index
        if os.path.exists(index_path):
            try:
                # Use the actual embedding model (OllamaEmbeddings) used in Model
                vectorstore = FAISS.load_local(
                    index_path, 
                    self.model.get_embedding_model(),
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing FAISS index from {index_path}")
            except Exception as e:
                print(f"Warning: Could not load existing index ({e}). Starting fresh.")
                vectorstore = None
                processed_ids = set() # Reset if index is gone

        total_chunks = len(chunks)
        print(f"Total chunks to process: {total_chunks}")
        
        for i, chunk in enumerate(chunks):
            if i in processed_ids:
                continue

            print(f"Processing Chunk {i + 1}/{total_chunks}...")
            
            # Using chunk.page_content logic
            facts, embeddings = self.process_chunk(chunk.page_content, i)
            
            if facts and embeddings:
                # Prepare data for FAISS
                # FAISS expects list of (text, vector) if using from_embeddings/add_embeddings?
                # Actually LangChain FAISS add_embeddings expects:
                # text_embeddings: Iterable[Tuple[str, List[float]]]
                
                text_embeddings = list(zip(facts, embeddings))
                metadatas = [{"chunk_id": i, "source": self.novel_name} for _ in facts]
                
                if vectorstore is None:
                    vectorstore = FAISS.from_embeddings(
                        text_embeddings=text_embeddings,
                        embedding=self.model.get_embedding_model(),
                        metadatas=metadatas
                    )
                else:
                    vectorstore.add_embeddings(
                        text_embeddings=text_embeddings,
                        metadatas=metadatas
                    )
                
                # Save regularly (e.g. every chunk)
                vectorstore.save_local(index_path)
            
            # Mark as processed
            processed_ids.add(i)
            save_cache(list(processed_ids), tracker_file)
            
        print("\nAll chunks processed.")

    def process_failed_chunks(self, output_api_error_indices: list[int]):
         # We want to process specifically these chunks.
        print(f"Retrying failed chunks: {output_api_error_indices}")
        if not os.path.exists("cache"):
            os.makedirs("cache")
            
        chunks_cache = f"cache/cache_chunks_{self.novel_name}.json"
        if os.path.exists(chunks_cache):
            print(f"Loading chunks from cache: {chunks_cache}")
            chunks = load_cache(chunks_cache)
        else:
            print("Chunks cache not found. Re-splitting book.")
            chunks_objs = self.split_book_into_chunks(chunk_size=10000)
            chunks = [c.page_content for c in chunks_objs]

        if not chunks:
            print("No chunks to process.")
            return

        index_path = f"{self.novel_name}_faiss_index"
        vectorstore = None
        
        # Try loading existing index
        if os.path.exists(index_path):
            try:
                vectorstore = FAISS.load_local(
                    index_path, 
                    self.model.get_embedding_model(),
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing FAISS index from {index_path}")
            except Exception as e:
                print(f"Warning: Could not load existing index ({e}). Starting fresh.")
                vectorstore = None
        
        for i in output_api_error_indices:
            if i < 0 or i >= len(chunks):
                print(f"Chunk ID {i} out of range. Skipping.")
                continue
                
            print(f"Processing Chunk {i}...")
            # Ideally we force re-processing even if cache exists, 
            # OR we assume the user only passes IDs that actually failed (so no cache or empty).
            # If we want to force explicit retry, we might need to delete cache first.
            chunk_text = chunks[i]
            
            # Let's delete the 'failed' cache if it exists and is empty/invalid to ensure clean retry
            facts_cache = f"cache/facts_{self.novel_name}_{i}.json"
            if os.path.exists(facts_cache):
                 # Optional: check if empty? For now, we trust process_chunk logic 
                 # or we assume the user knows it failed.
                 pass

            facts, embeddings = self.process_chunk(chunk_text, i)
            
            if facts and embeddings:
                text_embeddings = list(zip(facts, embeddings))
                metadatas = [{"chunk_id": i, "source": self.novel_name} for _ in facts]
                
                if vectorstore is None:
                    vectorstore = FAISS.from_embeddings(
                        text_embeddings=text_embeddings,
                        embedding=self.model.get_embedding_model(),
                        metadatas=metadatas
                    )
                else:
                    vectorstore.add_embeddings(
                        text_embeddings=text_embeddings,
                        metadatas=metadatas
                    )
                
                vectorstore.save_local(index_path)
                print(f"Chunk {i} successfully processed and saved.")
            else:
                 print(f"Chunk {i} failed again.")
