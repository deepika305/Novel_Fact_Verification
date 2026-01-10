from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSequence
import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from custom_llm import CustomLLM

# from dotenv import load_dotenv

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class Model:
    def __init__(self):
        # self.llm = ChatOllama(model="gemma3:4b", temperature=0)
        self.llm = CustomLLM()
        self.embedding_model = OllamaEmbeddings(model="embeddinggemma:latest")
    def get_llm(self):
        return self.llm
    def get_embedding_model(self):
        return self.embedding_model

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
    def __init__(self, embeddings_list=None):
        self.embeddings_list = embeddings_list or []

    def embed_documents(self, texts):
        # not used since vectors already exist
        return self.embeddings_list[:len(texts)]

    def embed_query(self, text):
        raise NotImplementedError("Use similarity_search_by_vector()")

def extract_and_embed_claims_langchain(claim_sentence: str, bookname: str, model):
        # if not GOOGLE_API_KEY:
        #     print("API key not set. Generating mock facts and embeddings.")
        #     # Mock fact extraction
        #     mock_facts = [f"Mock fact 1 from: {claim_sentence[:20]}...",
        #                 f"Mock fact 2 from : {claim_sentence[20:40]}..."]

        #     # Mock embedding generation (e.g., random vectors)
        #     # A common embedding dimension for Gemini models is 768 or 1536.
        #     # Using 768 as a placeholder for mock data consistency.
        #     mock_embedding_vectors = [np.random.rand(768).tolist() for _ in mock_facts]
        #     print(f"Mock extracted facts: {mock_facts}")
        #     return []

        # try:
            # Initialize the Gemini LLM for fact extraction using Langchain
            # Changed model from 'gemini-pro' to 'gemini-pro-latest' as it is available.






            # cache hatanahai
            # if os.path.exists(f"cache/cache_claims_{bookname}.json"):
            #     print("Loading from cache...")
                # extracted_text = load_cache(f"cahe/cache_claims_{bookname}.json")
            # else:
            llm = model.get_llm()

            # Define the prompt for fact extraction
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", (
                    "Extract distinct, atomic facts from the following sentences."
                    " List each fact on a new line as a sentence. Do not include introductory phrases."
                    " If no facts can be extracted, return an empty string."
                )),
                ("human", "Sentence: {sentence}")
            ])

            # Create a Langchain chain for claim extraction
            claim_extraction_chain = ({"sentence": RunnablePassthrough()} | prompt_template | llm | StrOutputParser())

            # Invoke the chain to extract claims
            extracted_text = claim_extraction_chain.invoke(claim_sentence)
            extracted_text = extracted_text.strip()
                # save_cache(extracted_text, f"cache/cache_claims_{bookname}.json")






            if not extracted_text:
                print("No claims extracted.")
                return []

            # Assuming claims are newline-separated
            claim_sentences = [claim.strip() for claim in extracted_text.split('\n') if claim.strip()]

            if not claim_sentences:
                print("No valid claim sentences parsed.")
                return []

            print(f"Extracted claims: {claim_sentences}")









            # Initialize the embedding model using Langchain
            # if os.path.exists(f"cache_embeddings_{bookname}.json"):
            #     print("Loading embeddings from cache...")
            #     embedding_vectors = load_cache(f"cache_embeddings_{bookname}.json")
            #     # embeddings_model = PrecomputedEmbeddings(embeddings_list)
            # else:
            embeddings_model = model.get_embedding_model()

            # Generate embeddings for each fact sentence
            embedding_vectors = embeddings_model.embed_documents(claim_sentences)
                # save_cache(embedding_vectors, f"cache_embeddings_{bookname}.json")






            list_of_documents = []
            for i, claim in enumerate(claim_sentences):
                doc = Document(page_content=claim, metadata = {"embedding": embedding_vectors[i]})
                list_of_documents.append(doc)

            return list_of_documents

        # except Exception as e:
        #     print(f"An error occurred during fact extraction or embedding: {e}")
        #     return []

def check_consistency(backstory: str, chunk: str, char: str, model):
    prompt_template = PromptTemplate(
    input_variables=["character_name", "backstory", "chunk"],
    template="""
              You are a literary consistency analyst.

              Your task is to determine whether the following text chunk from a novel
              is consistent with the given character backstory.

              Character Name:
              {character_name}

              Character Backstory:
              {backstory}

              Novel Chunk:
              {chunk}

              Instructions:
              - Carefully reason step by step.
              - Check for contradictions in personality, history, abilities, or motivations.
              - If the chunk aligns with the backstory, mark it as Consistent.
              - If it conflicts, mark it as Contradict.
              - Respond ONLY in valid JSON format.

              Required JSON format:
              {{
                "Verdict": "Consistent" or "Contradict",
                "Reason": "Clear 1 line explanation of your reasoning"
              }}
            """
            )
    output_parser = JsonOutputParser()
    safe_parser = OutputFixingParser.from_llm(
        parser=output_parser,
        llm=Model.get_llm(model)
    )
    chain = prompt_template | model.get_llm() | output_parser

    result = chain.invoke({
        "character_name": char,
        "backstory": backstory,
        "chunk": chunk
    })

    print(f"Consistency Check Result: {result}")

    return result

def dummy_function(bookname, char, content, model):
    print("dummy function called")
    backstory = f"This line is for {char} character: {content}"
    list_of_documents = extract_and_embed_claims_langchain(backstory, bookname, model)
    vectorstore = FAISS.load_local(
    f"{bookname}_faiss_index",
    embeddings=PrecomputedEmbeddings(),
    allow_dangerous_deserialization=True
    )
    print("vector store loaded...")
    final_verdict = "consistent"
    reason = "All claims are consistent with the backstory."
    for claim_doc in list_of_documents:
         query_emb = claim_doc.metadata["embedding"]
         query_vector = np.array(query_emb, dtype="float32")
         fact_docs = vectorstore.similarity_search_by_vector(query_vector, k=3)
         print("Similarity search done!!!")
         chunk_id = fact_docs[0].metadata['chunk_id']

         with open(f"cache/cache_chunks_{bookname}.json", "r") as f:
             chunks = json.load(f)

         chunk = chunks[int(chunk_id)]
         print("starting comaparison....")
         verdict = check_consistency(backstory, chunk, char, model)
         print("verdict calculated")
         if verdict["Verdict"].lower() == "contradict":
                final_verdict = "contradict"
                reason = verdict["Reason"]
                return final_verdict, reason
    print(f"Final Verdict: {final_verdict}, Reason: {reason}")
    return final_verdict, reason



