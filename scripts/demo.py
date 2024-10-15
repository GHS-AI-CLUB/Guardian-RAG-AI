import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path='./chorma_store')
collection = client.get_collection("ghs_embeddings")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

st.title("Gaurdian AI Demo")
query = st.text_input("Enter a question", "what are the computer science coureses offered here?")


if query: 
 with st.spinner("Loading . . . "): 

    query_text = "what is are the computer science at this school?"

    query_embedding = embedding_model.encode([query_text], convert_to_tensor=False)[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5 
    )

    for i, result in enumerate(results['documents']):
        print(f"Result {i + 1}:")
        print(f"Document: {result[0]}")
        context = result[0]


    template = """You are an assitant for Glastonbury High School. Answer the question below in a nice and proffessional manner. 
    Here is the conversation history {context}
    Question: {question}
    Answer:
    """

    model = OllamaLLM(model='llama3')
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    result = chain.invoke({'context': context, 'question': query_text})
    st.write(result)
