# Novel Fact Verification

Short guide to set up and run the Novel_Fact_Verification project.

## Project layout

- `Novel_Fact_Verification/` — main package containing source code and modules:
  - `backstory_loader.py`- modules
  - `custom_llm.py`- Wrapper for langchain LLM model 
  - `helper.py`- modules
  - `main.py` — primary entry/runner
  - `preprocessor.py` / `preprocessor_helper.py`- Preprocessor for Novel handling
  - `requirement.txt` — Python dependencies
  - `cache/` — saved embeddings, chunks and json caches
  - `<book>_faiss_index/` — vector store containing embedding vectors

## Requirements

- Python 3.9+ (3.10/3.11 recommended)
- On Windows, use PowerShell or CMD. For large datasets, ensure enough RAM or use memory-mapped operations.


## Setup (Windows)

1. Open bash and create a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\Activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r ./requirement.txt
```
***
## Commands to run the project (Runnable files)

- Run the main application : main.py

- If you want to add a new test.csv file, upload it inside Dataset folder ,then run the command given below.
- This command generates a result.csv file 

```bash
python ./main.py
```
***

# Pathway supported file
- If you want to use the pathway supported file
- You have to run the pw_server.py file
- This will Create a server that will continiously listen to input.csv
- You must paste the input csv values row-wise or directly multple rows (Do not type it in)
- The output will be updated in real-time inside output.csv

```bash
python ./pw_server.py
```
***

IMPORTANT
Do not run the preprocessor file if you want to get outputs for the given books as vector stores are already in the repository.

If you want to add a new book, upload it inside Dataset/Books folder.After that you need to run preprocessor.py file, you will have to get free llm access(tutorial given later).

```bash
python ./preprocessor.py
```

# How to run the LLM

There are different ways-

The best way is trying it locally: using `Ollama` and a strong model like `deepseek-r1:14b` 

for that download Ollama from https://ollama.com/download

then run `ollama pull deepseek-r1:14b` 

now We got the model 

now update the `custom_llm.py`  file as per your model name in this portion

```python
def __init__(self):
        # self.llm = ChatOllama(model="gemma3:4b", temperature=0)
        self.llm = CustomLLM()
        self.embedding_model = OllamaEmbeddings(
             model="embeddinggemma:latest"
             )
```

But if a high end GPU is not available then we can go for vps providers,

among free options there is Gemini API which is very easily rate limited in free tier, specially for a case of running `preprocessor.py` 

So we went for this approach `Cloudflare Workers AI` 

checkout the docs here : https://developers.cloudflare.com/workers-ai/

using their docs we made up a llm server 

This blog can also be used https://high-entropy-alloy.notion.site/getting-a-free-llm-ai-service

which takes some prompt in request and gives back the final result

and in the Model we used `gpt-oss-20b` 

And for the embedding model we used the `embeddinggemma` model with ollama.




