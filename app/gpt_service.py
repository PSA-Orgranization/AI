import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI
from .models import *
from .models import get_db_cursor
from pathlib import Path

client = OpenAI(api_key="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")

BASE_DIR = Path(__file__).parent.parent


def openai_chat_completion(prompt: str, model="gpt-4.1"):
  response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system" ,"content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
    max_tokens=512,
)
  return response.choices[0].message.content.strip()

def load_multiple_pdfs(folder_path: str):
    print(folder_path)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=2000, chunk_overlap=20
    )
    all_chunks, metadata_list = [], []
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    for file_name in file_list:
        reader = PdfReader(os.path.join(folder_path, file_name))
        for page_num, page in enumerate(reader.pages):
            raw_text = page.extract_text()
            if not raw_text:
                continue
            chunks = splitter.split_text(raw_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                metadata_list.append({
                    "source": file_name,
                    "page": page_num + 1,
                })
    return all_chunks, metadata_list

def create_vector_db_with_metadata(docs, metadata, collection_name="multi-pdf-collection"):
    embedding_function = SentenceTransformerEmbeddingFunction()
    client = chromadb.Client()
    collection = client.create_collection(
        name=collection_name, embedding_function=embedding_function, get_or_create=True
    )
    ids = [str(i) for i in range(len(docs))]
    collection.add(ids=ids, documents=docs, metadatas=metadata)
    return collection

def is_query_in_scope(query: str, history: list = None) -> bool:

    # 1) Quick keyword-based override:
    q_lower = query.lower()
    if "what to study next" in q_lower or "after studying" in q_lower or "study plan" in q_lower:
        return True

    # 2) Build the flattened chat history text for context
    history_text = ""
    if history:
        for i, turn in enumerate(history):
            role = "User" if i % 2 == 0 else "Assistant"
            history_text += f"{role}: {turn}\n"

    # 3) LLM prompt with expanded scope
    prompt = f"""
You are a smart query classifier for a Competitive Programming Assistant.

Your task is to determine whether the following query is reasonably related to:
- Competitive programming problems & solutions
- Algorithms & data structures
- Study plans or “what to study next” roadmaps for competitive programming
- Difficulty levels (Bronze, Silver, Gold, etc.)
- Programming contests or coding platforms (Codeforces, LeetCode, etc.)
- Problem‑solving strategies and practice advice
- or anything re;ated to study of the competitive programming or roadmaps or content.
If the query is even partially about any of these, reply ONLY with:
IN_SCOPE

If it is clearly unrelated—personal life, general knowledge, history, science—reply ONLY with:
OUT_OF_SCOPE

Do not explain. Do not give examples. Return exactly one word.

Chat History:
{history_text}

User Query:
{query}
""".strip()

    # 4) Call your chat-completion helper
    result = openai_chat_completion(prompt)

    # 5) (Optional) debug prints
    print("🔍 Classifier Prompt:\n", prompt)
    print("🔍 Classifier Result:", result.strip())

    # 6) Return True only if the model says IN_SCOPE
    return result.strip().upper() == "IN_SCOPE"

def augment_query(query: str):
    prompt = f"""
You are a Competitive Programming Assistant. Your task is to improve a user's question by generating a few related or rephrased queries that help retrieve more relevant technical content from a knowledge base.

Only generate queries that are strictly about:
- Algorithms
- Data Structures
- Competitive Programming techniques or strategies

Original User Query:
"{query}"

Please generate 3 to 5 concise, related or rephrased versions of this query. Do not explain anything, just list them.
"""
    output = openai_chat_completion(prompt)
    return [line.strip("-• ").strip() for line in output.split("\n") if line.strip()]

def retrieve_documents_with_metadata(collection, queries, n_results=5):
    results = collection.query(query_texts=queries, n_results=n_results, include=["documents", "metadatas"])
    combined = []
    for docs, metas in zip(results["documents"], results["metadatas"]):
        for doc, meta in zip(docs, metas):
            combined.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "?")
            })
    seen, unique = set(), []
    for item in combined:
        if item["content"] not in seen:
            seen.add(item["content"])
            unique.append(item)
    return unique

def generate_response_with_rag_context_ROADMAP(user_input, session_id=None, username="default_user", email="default@example.com"):
    cursor = get_db_cursor()
    chunks_roadmap, metadata_roadmap = load_multiple_pdfs(str(BASE_DIR / "docs"))
    collection_roadmap = create_vector_db_with_metadata(chunks_roadmap, metadata_roadmap)

    try:
        # Step 1: Manage user and session
        user_id = fetch_or_insert_user(cursor, username, email)
        if not user_id:
            return "Error: Could not create or retrieve user"

        if session_id is None or not check_session_exists(cursor, session_id):
            session_id = fetch_or_insert_session(cursor, user_id)
            if not session_id:
                return "Error: Could not create or retrieve session"

        # Step 2: Retrieve chat history from DB
        chat_history = retrieve_chat_history(cursor, session_id)

        # Convert flat chat_history list to alternating history
        flat_history = []
        for record in chat_history:
            if len(record) >= 2:
                flat_history.append(record[0])  # user prompt
                flat_history.append(record[1])  # assistant response

        # Step 3: Scope check
        if not is_query_in_scope(user_input, flat_history):
            return "Sorry, I am a Competitive Programming Assistant. This question falls outside my scope. Please ask about coding or competitive programming topics only."

        # Step 4: Augment query and retrieve documents
        augmented = augment_query(user_input)
        queries = [user_input] + augmented
        retrieved_docs = retrieve_documents_with_metadata(collection_roadmap, queries)

        # Step 5: Build context string
        context = "\n\n".join(
            f"[{doc['source']}, page {doc['page']}] {doc['content']}" for doc in retrieved_docs
        )

        # Step 6: Construct full prompt with memory
        messages = [{"role": "system", "content": """
You are a highly intelligent and efficient Problem-Solving Assistant specialized in data structures, algorithms, and competitive programming.
You use C++ as your primary language.

Your main goal is to help users solve algorithmic and competitive programming problems effectively.

Focus especially on:
- Carefully analyzing the constraints, as they guide the choice of algorithm (brute force, greedy, DP, etc.)
- Understanding the input and output formats and providing code that correctly handles edge cases
- Identifying time and space complexity limits based on constraints
- Ensuring solutions scale within the given limits

Always provide:
1. A clear explanation of the problem
2. A breakdown of constraints, input/output format, and what they imply
3. A step-by-step plan to solve the problem
4. Clean, well-commented C++ code
5. Time and space complexity analysis
6. Additional tips if the problem is relevant for competitive programming

Keep explanations beginner-friendly but concise. Use standard C++ STL and best practices for coding and performance."""}]
        for i, turn in enumerate(flat_history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})

        # Add context and final user question
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})

        # Step 7: Call OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

        reply = completion.choices[0].message.content.strip()

        # Step 8: Store chat in DB
        store_chat_message(cursor, session_id, user_input, reply)
        cursor.connection.commit()

        return reply

    except Exception as e:
        print(f"Error in generate_response_with_rag_context: {e}")
        return f"Error occurred: {str(e)}"

def generate_response_with_rag_context_General(user_input, session_id=None, username="default_user", email="default@example.com"):
    cursor = get_db_cursor()

    chunks_general, metadata_general = load_multiple_pdfs(str(BASE_DIR / "docs2"))
    collection_general = create_vector_db_with_metadata(chunks_general, metadata_general)

    try:
        # Step 1: Manage user and session
        user_id = fetch_or_insert_user(cursor, username, email)
        if not user_id:
            return "Error: Could not create or retrieve user"

        if session_id is None or not check_session_exists(cursor, session_id):
            session_id = fetch_or_insert_session(cursor, user_id)
            if not session_id:
                return "Error: Could not create or retrieve session"

        # Step 2: Retrieve chat history from DB
        chat_history = retrieve_chat_history(cursor, session_id)

        # Convert flat chat_history list to alternating history
        flat_history = []
        for record in chat_history:
            if len(record) >= 2:
                flat_history.append(record[0])  # user prompt
                flat_history.append(record[1])  # assistant response

        # Step 3: Scope check
        if not is_query_in_scope(user_input, flat_history):
            return "Sorry, I am a Competitive Programming Assistant. This question falls outside my scope. Please ask about coding or competitive programming topics only."

        # Step 4: Augment query and retrieve documents
        augmented = augment_query(user_input)
        queries = [user_input] + augmented
        retrieved_docs = retrieve_documents_with_metadata(collection_general, queries)

        # Step 5: Build context string
        context = "\n\n".join(
            f"[{doc['source']}, page {doc['page']}] {doc['content']}" for doc in retrieved_docs
        )

        # Step 6: Construct full prompt with memory
        messages = [{"role": "system", "content": """
You are a highly intelligent and efficient Problem-Solving Assistant specialized in data structures, algorithms, and competitive programming.
You use C++ as your primary language.

Your main goal is to help users solve algorithmic and competitive programming problems effectively.

Focus especially on:
- Carefully analyzing the constraints, as they guide the choice of algorithm (brute force, greedy, DP, etc.)
- Understanding the input and output formats and providing code that correctly handles edge cases
- Identifying time and space complexity limits based on constraints
- Ensuring solutions scale within the given limits

Always provide:
1. A clear explanation of the problem
2. A breakdown of constraints, input/output format, and what they imply
3. A step-by-step plan to solve the problem
4. Clean, well-commented C++ code
5. Time and space complexity analysis
6. Additional tips if the problem is relevant for competitive programming

Keep explanations beginner-friendly but concise. Use standard C++ STL and best practices for coding and performance."""}]
        for i, turn in enumerate(flat_history):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn})

        # Add context and final user question
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})

        # Step 7: Call OpenAI API
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )

        reply = completion.choices[0].message.content.strip()

        # Step 8: Store chat in DB
        store_chat_message(cursor, session_id, user_input, reply)
        cursor.connection.commit()

        return reply

    except Exception as e:
        print(f"Error in generate_response_with_rag_context: {e}")
        return f"Error occurred: {str(e)}"




