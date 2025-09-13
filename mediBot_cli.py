import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# --- Setup ---
try:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error during setup: {e}")
    exit()

# --- Load global resources ---
print("Loading resources...")
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="db")
    collection = client.get_collection(name="medical_qa")
    print("Resources loaded successfully.")
except Exception as e:
    print(f"Error loading resources: {e}")
    exit()


def get_bot_response(user_query: str, model) -> str:
    """
    Generates a response from the chatbot using a RAG approach with a fallback.
    """
    # Retrieval and Relevance Check
    query_embedding = model.encode(user_query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["documents", "distances"]
    )
    
    context_documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    RELEVANCE_THRESHOLD = 0.6 
    
    # RAG Path (if context is relevant)
    if context_documents and distances[0] < RELEVANCE_THRESHOLD:
        context = "\n\n---\n\n".join(context_documents)
        prompt_template = f"""
        You are a helpful medical assistant. Your role is to provide a clear and concise summary 
        (in 2-3 sentences) based ONLY on the following context.

        CONTEXT:
        {context}

        QUESTION:
        {user_query}

        ANSWER:
        """
        try:
            llm = genai.GenerativeModel("gemini-1.5-flash")
            response = llm.generate_content(prompt_template)
            return response.text.strip()
        except Exception as e:
            return f"An error occurred while generating the response: {e}"
    
    # Fallback Path (if no relevant context is found)
    else:
        fallback_prompt = f"""
        You are a helpful medical assistant with a limited knowledge base. The user has asked a question that is not in your documents: "{user_query}"

        Follow these rules precisely:
        1. First, determine if the question is related to medicine, health, symptoms, or treatments.
        2. If it IS a medical question, provide a general, helpful answer but you MUST include this exact disclaimer at the end: "**Disclaimer:** This information is from a general AI and is not a substitute for professional medical advice. Out of the Knowledge base"
        3. If it is NOT a medical question, you MUST respond with only this exact phrase: "I am a medical assistant and can only answer questions related to health and medicine."

        Now, provide the correct response based on the user's question.
        """
        try:
            llm = genai.GenerativeModel("gemini-1.5-flash")
            response = llm.generate_content(fallback_prompt)
            return response.text.strip()
        except Exception as e:
            return f"An error occurred during the fallback process: {e}"


def main(model):
    """
    Runs the main command-line interface loop for the chatbot.
    """
    print("\n--- Medical FAQ Chatbot (CLI) ---")
    print("Ask a medical question. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
            
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break
        
        response = get_bot_response(user_input, model)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main(embedding_model)