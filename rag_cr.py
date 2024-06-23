"""
Source code for a basic RAG chain (the source being a YouTube video)
for a question-answer application (not a chatbot though)
"""


# SOURCE ENVIRONMENT
from dotenv import load_dotenv

load_dotenv()   # LLM, vectorstore API Keys


# CHOOSE AND LOAD
from langchain_community.document_loaders import YoutubeLoader

YOUTUBE_URL = "https://www.youtube.com/watch?v=cdiD-9MMpb0"

transcript_loader = YoutubeLoader.from_youtube_url(YOUTUBE_URL)
transcription = [transcript_loader.load()[0]]


# SPLIT
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1_000,
    chunk_overlap = 200
)
transcription_splits = text_splitter.split_documents(transcription)


# VECTORSTORE AND RETRIEVER
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents = transcription_splits,
    collection_name = "rag-chroma",
    embedding = OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


# PROMPT
from langchain_core.prompts import ChatPromptTemplate

prompt = """
Answer the question based on the context below. \
If you can't answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

# LLM
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(model="gpt-3.5-turbo", temperature=0)


## STRUCTURE OUTPUT
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()


# CHAIN COMPONENTS
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

# INVOKE
rag_chain.invoke("What is synthetic intelligence?")
