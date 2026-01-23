# 🩺 Medical Chatbot with RAG

A **Retrieval-Augmented Generation (RAG)** chatbot for general medical question answering, powered by **OpenAI GPT-5 nano** and built with **LangChain**. The system uses a **hybrid retrieval pipeline** to search trusted medical knowledge from *The Gale Encyclopedia of Medicine* and generate concise, contextual, and user-friendly answers.

This project demonstrates how to combine modern LLMs with vector databases and sparse retrieval to build production-style medical QA systems.

---

## ⚠️ Medical Disclaimer

> **IMPORTANT:** This chatbot is designed for **informational and educational purposes only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.

✅ This tool should be used as a **supplement**, not a replacement, for professional medical consultation.

---

## ✨ Key Features

* **🚀 Fast & Cost-Effective LLM**
  Powered by **OpenAI GPT-5 nano** for concise, efficient responses.

* **🧠 Hybrid Retrieval (Dense + Sparse)**
  Combines semantic embeddings with keyword-based search for higher accuracy using a tunable `alpha` score.

* **📚 Trusted Medical Knowledge Base**
  Built from *The Gale Encyclopedia of Medicine*.

* **🧩 Smart Chunking**
  Uses LangChain’s `RecursiveCharacterTextSplitter` with medical-aware chunking strategy.

* **🔍 Vector Search with Pinecone**
  Stores and retrieves medical documents efficiently at scale.

* **🖥️ Interactive UI**
  Simple and clean **Gradio** interface for real-time medical Q&A.

* **🔗 LangChain Orchestration**
  Seamlessly integrates LLMs, retrievers, embeddings, and memory.

* **🗂️ Session-Based Conversational Memory**
  Maintains conversation context per user session using `langgraph.checkpoint.memory.InMemorySaver`, with the Gradio `session_id` passed as the LangGraph `thread_id` to ensure stable, multi-turn interactions.

---

## 🏗️ Architecture Overview

```
User Query
   ↓
Gradio Interface
   ↓
LangChain Orchestrator
   ↓
Hybrid Retriever (Dense + Sparse)
   ↓
Pinecone Vector Database
   ↓
Context Assembly
   ↓
OpenAI GPT-5 nano
   ↓
Answer to User
```

---

## 🔧 Tech Stack

* **LLM:** OpenAI GPT-5 nano
* **Framework:** LangChain
* **Vector DB:** Pinecone
* **Embeddings:**

  * `text-embedding-3-large` (dense)
  * `pinecone-sparse-english-v0` (sparse)
* **Interface:** Gradio
* **Language:** Python

---

## 🎮 Demo

You can interact with the chatbot via the **Gradio interface** for real-time medical Q&A.

Example queries:

* "What are the symptoms of diabetes?"
* "How is hypertension treated?"
* "What causes migraine headaches?"
* "Explain the difference between Type 1 and Type 2 diabetes."

---

## ✅ Prerequisites

* Python **3.10+** (tested on **3.12.9**)
* OpenAI API key
* Pinecone API key

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/medical-qa-chatbot.git
cd Medical-Chatbot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

---

### 5️⃣ Configure the System

Edit `config.yaml`:

```yaml
index_name: your_index_name
name_space: your_namespace
```

Feel free to adjust other parameters such as chunk size, retriever alpha, and model settings.

---

### 6️⃣ Customize the System Prompt

You can modify the behavior of the assistant in:

```
system_prompt.txt
```

This controls tone, safety, and answer formatting.

---

## 💻 Usage

Start the chatbot:

```bash
python gradioapp.py
```

The Gradio interface will launch at:

```
http://localhost:7860
```

---

## ⚠️ Limitations & Considerations

### 🔧 Technical

* Answers are limited to content in *The Gale Encyclopedia of Medicine*.
* May not reflect the most recent clinical guidelines.
* Performance depends on query clarity and retrieved context quality.

### ⚖️ Ethical

* Not suitable for emergency medical situations.
* Should not be used for self-diagnosis or treatment decisions.
* Possible biases from source material.
* Human medical professional oversight is essential.
---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to fork the project and submit a PR.

---

## ⭐ Acknowledgments

* OpenAI
* LangChain
* Pinecone
* The Gale Encyclopedia of Medicine

---

Happy building! 🚀
