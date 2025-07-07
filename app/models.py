import sqlite3  

conn = sqlite3.connect('psa.sqlite' , check_same_thread=False) 
cursor = conn.cursor()

def get_db_cursor():
    return conn.cursor()

def insert_user(cursor, username, email):
    try:
        cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", (username, email))
        return cursor.lastrowid
    except Exception as e:
        print(f"Error inserting user: {e}")
        return None

def fetch_or_insert_user(cursor, username, email):
    cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
    result = cursor.fetchone()
    if result:
        return result[0]
    return insert_user(cursor, username, email)

def insert_session(cursor, user_id, session_name):
    try:
        cursor.execute("INSERT INTO sessions (user_id, session_name) VALUES (?, ?)", (user_id, session_name))
        return cursor.lastrowid
    except Exception as e:
        print(f"Error inserting session: {e}")
        return None

def fetch_or_insert_session(cursor, user_id, session_name="Default Session"):
    cursor.execute("SELECT id FROM sessions WHERE user_id = ? ORDER BY id DESC", (user_id,))
    result = cursor.fetchone()
    if result:
        return result[0]
    return insert_session(cursor, user_id, session_name)

def store_chat_message(cursor, session_id, prompt, response):
    try:
        cursor.execute(
            "INSERT INTO chat_memory (session_id, prompt, response) VALUES (?, ?, ?)",
            (session_id, prompt, response)
        )
    except Exception as e:
        print(f"Error storing chat message: {e}")

def retrieve_chat_history(cursor, session_id):
    try:
        cursor.execute(
            "SELECT prompt, response, timestamp FROM chat_memory WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        return cursor.fetchall()
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def check_session_exists(cursor, session_id):
    try:
        cursor.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,))
        return cursor.fetchone()[0] > 0
    except Exception as e:
        print(f"Error checking session existence: {e}")
        return False
