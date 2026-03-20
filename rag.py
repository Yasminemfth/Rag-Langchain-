import os
from dotenv import load_dotenv
load_dotenv()

import bs4
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

# Modèle via OpenRouter
model = ChatOpenAI(
    model="anthropic/claude-3-haiku",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)

# Embeddings gratuits
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store en mémoire
vector_store = InMemoryVectorStore(embeddings)

# Charger le portfolio
print("📄 Chargement du portfolio...")
loader = WebBaseLoader(
    web_paths=(
        "https://portfolio-yasmine-meftah.onrender.com/",
        "https://portfolio-yasmine-meftah.onrender.com/about",
        "https://portfolio-yasmine-meftah.onrender.com/projects",
    ),
)
docs = loader.load()

# Découper en morceaux
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
vector_store.add_documents(chunks)
print(f"✅ {len(chunks)} chunks indexés")

# Boucle de questions interactives
print("\n💬 Pose tes questions sur le portfolio (tape 'exit' pour quitter)\n")

while True:
    question = input("❓ Ta question : ")
    if question.lower() == "exit":
        break

    docs_trouves = vector_store.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs_trouves)

    response = model.invoke([
        SystemMessage(content=f"Tu es un assistant qui répond aux questions sur le portfolio de Yasmine Meftah. Réponds en français en te basant uniquement sur ce contexte :\n\n{context}"),
        HumanMessage(content=question)
    ])

    print("💬 Réponse :", response.content)
    print()