import os
import pandas as pd 
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
 

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 1: Convert PDF to text
import textract

doc = textract.process("./Docs/agreement.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open("./Docs/agreement.txt", "w", encoding="utf-8") as f:
    f.write(doc.decode("utf-8"))


with open("./Docs/agreement.txt", "r") as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0])

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

query = "Who created transformers?"
docs = db.similarity_search(query)
docs[0]

# Create QA chain to integrate similarity search with user queries (answer query from knowledge base)

chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")

query = ""
docs = db.similarity_search(query)

chain.run(input_documents=docs, question=query)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

chat_history = []


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data["question"]
    if query.lower() == "exit":
        return {"chat_history": chat_history}
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return {"chat_history": chat_history}
 