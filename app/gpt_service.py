from openai import OpenAI
from app.models import *
from app.db_conn import get_db_cursor

client = OpenAI(api_key="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")

system_prompt = """
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

Keep explanations beginner-friendly but concise. Use standard C++ STL and best practices for coding and performance.
"""

def generate_response_with_context(user_input, session_id=None, username="default_user", email="default@example.com"):
    cursor = get_db_cursor()
    try:
        user_id = fetch_or_insert_user(cursor, username, email)
        if not user_id:
            return "Error: Could not create or retrieve user"

        if session_id is None or not check_session_exists(cursor, session_id):
            session_id = fetch_or_insert_session(cursor, user_id)
            if not session_id:
                return "Error: Could not create or retrieve session"

        chat_history = retrieve_chat_history(cursor, session_id)

        messages = [{"role": "system", "content": system_prompt}]
        for record in chat_history:
            if len(record) >= 2:
                messages.append({"role": "user", "content": record[0]})
                messages.append({"role": "assistant", "content": record[1]})

        messages.append({"role": "user", "content": user_input})

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            top_p=1,
            max_tokens=2048,
        )

        reply = completion.choices[0].message.content
        store_chat_message(cursor, session_id, user_input, reply)
        cursor.connection.commit()
        return reply

    except Exception as e:
        print(f"Error in generate_response_with_context: {e}")
        return f"Error occurred: {str(e)}"
