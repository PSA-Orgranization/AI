import sqlite3  # Replaces pyodbc

# Connect to SQLite database (creates if doesn't exist)
conn = sqlite3.connect('psa.sqlite' , check_same_thread=False)  # Database stored in psa.sqlite file
cursor = conn.cursor()

def get_db_cursor():
    return conn.cursor()