import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=DESKTOP-V87H8C1;"
    "DATABASE=PSA;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

def get_db_cursor():
    return conn.cursor()