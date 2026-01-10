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
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
        with open("data.json", "w") as f:
            json.dump(self.book_chunks, f)
        return chunks
    
    # method to extract facts from chunks
    def extract_and_embed_facts_langchain(self, claim_sentence: str, chunk_id: int):
        if not GOOGLE_API_KEY:
            print("API key not set. Generating mock facts and embeddings.")
            # Mock fact extraction
            mock_facts = [f"Mock fact 1 from chunk {chunk_id}: {claim_sentence[:20]}...",
                        f"Mock fact 2 from chunk {chunk_id}: {claim_sentence[20:40]}..."]

            # Mock embedding generation (e.g., random vectors)
            # A common embedding dimension for Gemini models is 768 or 1536.
            # Using 768 as a placeholder for mock data consistency.
            mock_embedding_vectors = [np.random.rand(768).tolist() for _ in mock_facts]
            print(f"Mock extracted facts: {mock_facts}")
            return {chunk_id: mock_embedding_vectors}

        try:
            # Initialize the Gemini LLM for fact extraction using Langchain
            # Changed model from 'gemini-pro' to 'gemini-pro-latest' as it is available.
            if os.path.exists(f"cache/cache_facts_{self.novel_name}.json"):
                print("Loading from cache...")
                extracted_text = load_cache(f"cache/cache_facts_{self.novel_name}.json")
            else:
                llm = ChatOllama(model="gemma3:4b", temperature=0)

                # Define the prompt for fact extraction
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "Extract distinct, atomic facts from the following sentence. List each fact on a new line. Do not include introductory phrases or numbers. If no facts can be extracted, return an empty string."),
                    ("human", "Sentence: {sentence}")
                ])

                # Create a Langchain chain for fact extraction
                fact_extraction_chain = ({"sentence": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())

                # Invoke the chain to extract facts
                extracted_text = fact_extraction_chain.invoke(claim_sentence)
                extracted_text = extracted_text.strip()
                save_cache(extracted_text, f"cache/cache_facts_{self.novel_name}.json")

            if not extracted_text:
                print("No facts extracted.")
                return {chunk_id: []}

            # Assuming facts are newline-separated
            fact_sentences = [fact.strip() for fact in extracted_text.split('\n') if fact.strip()]

            if not fact_sentences:
                print("No valid fact sentences parsed.")
                return {chunk_id: []}

            print(f"Extracted facts: {fact_sentences}")

            # Initialize the embedding model using Langchain
            if os.path.exists(f"cache/cache_embeddings_facts_{self.novel_name}.json"):
                print("Loading embeddings from cache...")
                embedding_vectors = load_cache(f"cache/cache_embeddings_facts_{self.novel_name}.json")
                # embeddings_model = PrecomputedEmbeddings(embeddings_list)
            else:
                embeddings_model = OllamaEmbeddings(model="embeddinggemma:latest")
                # Generate embeddings for each fact sentence
                embedding_vectors = embeddings_model.embed_documents(fact_sentences)
                save_cache(embedding_vectors, f"cache/cache_embeddings_facts_{self.novel_name}.json")

            return {chunk_id: embedding_vectors}

        except Exception as e:
            print(f"An error occurred during fact extraction or embedding: {e}")
            return {chunk_id: []}

    def generate_embeddings(self):
        # forming the chain
        book_path = self.novel_path
        processing_chain = (
        RunnableLambda(lambda path: self.split_book_into_chunks(chunk_size=10000)) # First, split the book into chunks, using the updated chunk_size for consistency
        | RunnableLambda(lambda chunks: [
            self.extract_and_embed_facts_langchain(chunk.page_content, i)
            for i, chunk in enumerate(chunks)
        ]) # Then, extract facts and embeddings for each chunk
        | RunnableLambda(lambda list_of_dicts: {
            k: v for d in list_of_dicts for k, v in d.items()
        }) # Combine all chunk results into a single dictionary
        | RunnableLambda(lambda final_dict: json.dumps(final_dict, indent=2)) # Convert the final dictionary to a JSON string
        )

        # Invoke the chain (this might take a long time to run due to processing the entire book)
        print("Starting book processing chain...")
        print(book_path)
        result_json_string = processing_chain.invoke(book_path)
        print("Book processing completed.")
        return result_json_string
    
    # forward function
    def forward(self):
        print("Generating embeddings and storing in FAISS vector store...")
        embeddings_json = self.generate_embeddings()
        print("\n\nEmbeddings JSON generated!!!\n...")
        # store in vector store faiss with chunk id as key and embedding as value
        embeddings_dict = json.loads(embeddings_json)
        documents = []
        embeddings_list = []
        ids = []
        print("Preparing documents and embeddings for FAISS...")

        idx = 0
        for chunk_id, vectors in embeddings_dict.items():
            for vec in vectors:
                documents.append(
                    Document(
                        page_content=" ",   # dummy text (not used)
                        metadata={"chunk_id": chunk_id},
                        embedding=vec
                    )
                )
                embeddings_list.append(vec)
                # ids.append(f"{chunk_id}_{idx}")
                # idx += 1
        print(f"Total documents prepared: {len(documents)}")
        embedding_model = PrecomputedEmbeddings(embeddings_list)
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )
        print("FAISS vector store created.")
        vectorstore.save_local(self.novel_name + "_faiss_index")
        print(f"FAISS vector store saved as {self.novel_name}_faiss_index")