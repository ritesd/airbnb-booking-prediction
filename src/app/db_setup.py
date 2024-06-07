import sqlite3

def create_db():
    # Connect to the SQLite database (it will create a new database if it doesn't exist)
    conn = sqlite3.connect("my_database.db")

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Create a table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Commit the changes and close the database connection
    conn.commit()
    conn.close()


def create_test_db():
    # Connect to the SQLite database (it will create a new database if it doesn't exist)
    conn = sqlite3.connect("my_test_database.db")

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Create a table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Commit the changes and close the database connection
    conn.commit()
    conn.close()

if __name__=='__main__':
    #create_db()
    create_test_db()