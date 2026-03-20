# RAG LangChain — Portfolio Yasmine Meftah

Projet RAG (Retrieval Augmented Generation) réalisé avec l'assistance de Claude AI dans le cadre d'un apprentissage de l'IA et du traitement du langage naturel avec LangChain.

Le système charge les pages du portfolio de Yasmine Meftah, les indexe dans un vector store, puis répond à des questions en langage naturel en se basant uniquement sur le contenu récupéré.

## Fonctionnement

1. Les pages du portfolio sont chargées via `WebBaseLoader`
2. Le contenu est découpé en chunks de 1000 caractères
3. Chaque chunk est converti en vecteur via un modèle HuggingFace (sans clé API)
4. Les vecteurs sont stockés en mémoire avec `InMemoryVectorStore`
5. À chaque question, les chunks les plus pertinents sont récupérés et envoyés au LLM avec la question

## Stack technique

- Python 3.13
- LangChain — orchestration RAG
- LangChain OpenAI — connexion au LLM via OpenRouter
- HuggingFace Embeddings — modèle `all-MiniLM-L6-v2` (gratuit, local)
- OpenRouter API — accès au modèle `claude-3-haiku`

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-openai langchain-community langchain-text-splitters langchain-core bs4 python-dotenv sentence-transformers
```

## Configuration

Créer un fichier `.env` à la racine :

```
OPENAI_API_KEY=sk-or-v1-REMPLACE-MOI
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

La clé API OpenRouter est disponible sur [openrouter.ai/keys](https://openrouter.ai/keys).

## Utilisation

```bash
python rag.py
```

Le programme charge les pages du portfolio, les indexe, puis attend tes questions. Tape `exit` pour quitter.

Exemples de questions :
- "Comment s'appelle la personne du portfolio ?"
- "Quels sont ses projets ?"
- "Quelles sont ses compétences ?"
- "Quelle est sa formation ?"

## Structure

```
Rag-Langchain/
├── rag.py        — Script principal
├── .env          — Clé API (ne pas pusher sur GitHub)
├── .gitignore
└── README.md
```

## Licence

MIT