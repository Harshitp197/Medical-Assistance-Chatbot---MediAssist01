# Medical FAQ Chatbot - A RAG-based Approach

This project is a command-line chatbot designed to answer medical questions using a Retrieval-Augmented Generation (RAG) pipeline. The chatbot leverages a local knowledge base created from a medical Q&A dataset to ensure its answers are grounded in reliable information.

The system is built to be robust, handling both in-scope and out-of-scope questions intelligently with a hybrid approach and an automated fallback mechanism.

---

## Features

-   **RAG Pipeline**: Utilizes a local vector database (ChromaDB) and a sentence transformer model for efficient, relevant context retrieval.
-   **LLM Integration**: Uses the Google Gemini API (via `gemini-1.5-flash`) for natural language generation.
-   **Hybrid Logic**: If a question is within the knowledge base, it provides a context-grounded answer.
-   **Intelligent Fallback**: For out-of-scope questions, the bot classifies the user's intent to either answer general medical queries with a disclaimer or politely decline non-medical questions.
-   **Relevance Thresholding**: Prevents the retrieval of irrelevant documents, making the system more accurate.
-   **Dockerized Environment**: The entire application is containerized with Docker for easy, one-command execution and perfect reproducibility.

---

## Project Structure

```
medical-faq-chatbot/
│
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── build_database.py
├── mediBot_cli.py
├── train.csv
└── .env
```

---

## How to Run

There are two ways to run this project. The recommended method is using Docker.

### Option 1: Using Docker (Recommended)

This is the easiest way to run the project, as it handles all setup and dependencies automatically.

**1. Build the Docker Image**
Navigate to the project's root directory in your terminal and run:
```bash
docker build -t medical-chatbot .
```
This command creates a Docker image named `medical-chatbot`, installs all dependencies, and pre-builds the vector database.

**2. Run the Docker Container**
Run the chatbot with the following command. You must have your `.env` file with the `GEMINI_API_KEY` in the same directory.
```bash
docker run -it --rm --env-file .env medical-chatbot
```
The chatbot will start immediately.

---

### Option 2: Manual Setup

Follow these steps to run the project locally without Docker.

**1. Clone the Repository**
```bash
git clone <your-repository-url>
cd medical-faq-chatbot
```

**2. Create a Python Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set Up API Key**
You will need a free API key from **Google AI Studio**.
1. Create a file named `.env` in the project root.
2. Add your API key to the file as follows:
   ```
   GEMINI_API_KEY="your_google_api_key_here"
   ```

**5. Download the Dataset**
This project uses the "Comprehensive Medical Q&A Dataset" from Kaggle.
- **Download Link:** [Kaggle – Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/jpmiller/comprehensive-medical-q-a-dataset)
- Place the `train.csv` file in the root of the project directory.

**6. Build the Knowledge Base**
This script processes the dataset and creates the local vector database. It only needs to be run once.
```bash
python build_database.py
```

**7. Run the Chatbot**
Once the database is built, you can start the chatbot.
```bash
python mediBot_cli.py
```
To exit, type `exit`.

---

## Design Choices

Several key decisions were made to ensure the chatbot is robust, efficient, and reproducible.

1.  **Vector Database: ChromaDB**
    -   I chose **ChromaDB** because it's a lightweight, open-source vector database that is extremely easy to set up and manage locally. It handles both vector and metadata storage, simplifying the RAG pipeline significantly compared to lower-level libraries like FAISS.

2.  **Embedding Model: `all-MiniLM-L6-v2`**
    -   This `sentence-transformers` model was selected for its excellent balance of performance, speed, and size. It runs entirely locally, adhering to the "free tools" constraint, and is a standard, highly effective model for semantic search.

3.  **LLM: Google Gemini API (`gemini-1.5-flash`)**
    -   Initially, OpenAI's API was considered, but its free trial proved unreliable for new accounts. To ensure the project was functional under the "free tier" constraint, I pivoted to the **Google Gemini API**, which offers a generous and stable free tier. This switch demonstrates adaptability and a practical approach to problem-solving.

4.  **Hybrid RAG with Fallback Logic**
    -   A pure RAG system fails when a query is outside its knowledge base. To address this, I implemented a hybrid system. A **relevance threshold** on the retrieval step filters out irrelevant results. If no relevant context is found, a sophisticated fallback prompt instructs the LLM on how to act, making the chatbot significantly more intelligent and user-friendly.