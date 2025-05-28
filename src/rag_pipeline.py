from src.config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, CHROMA_PATH
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import re


def load_documents():
    # Load documents from a text file
    loader = TextLoader("data/dialogs.txt")
    documents = loader.load()
    return documents

def split_documents(documents):
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    return texts

def ingest_documents(texts):
    # Create a vector store from the documents
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_PATH)
    vector_store.persist()
    return vector_store

def load_vector_store():
    embendding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embendding)
    return vector_store

def get_conversational_chain():
    vectore_store = load_vector_store()
    hf_pipeline = pipeline("text-generation", model=LLM_MODEL_NAME, device=-1, max_new_tokens=200, do_sample=True, temperature=0.7, top_k=10, top_p=0.95, num_return_sequences=1)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2, return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectore_store.as_retriever(),
        memory=memory,
        condense_question_llm=llm,
        return_source_documents=False,
        verbose=False,
    )     

    return qa_chain   
        
qa_chain = get_conversational_chain()

def ask_question(question: str) -> dict:
    chat_history = qa_chain.memory.buffer if hasattr(qa_chain, "memory") else []
    result = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    

    raw_answer = result["answer"]

    answer_matches = re.findall(r"Helpful Answer:\s(.*)", raw_answer)
    if answer_matches: 
        final_answer = answer_matches[-1].strip()
    else:
        final_answer = raw_answer.strip()

    cleaned_history = []
    for msg in result["chat_history"]:
        cleaned_history.append({
            "type": msg.type,
            "content": msg.content,
        })

    return  {
        "qurstion": question,
        "answer": final_answer,
        "chat_history": cleaned_history}
   