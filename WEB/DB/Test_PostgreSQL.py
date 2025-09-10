from dotenv import load_dotenv
import os
import psycopg2
from psycopg2 import OperationalError

load_dotenv()

def create_connection(db_name, db_user, db_password, db_host='localhost', db_port=5432):
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        print("Connection to PostgreSQL established successfully")
        return conn
    except OperationalError as e:
        print(f"The error '{e}' occurred")
        return None

def execute_query(connection, query, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(query, params)
        connection.commit()
        print(f"Query executed successfully, {cursor.rowcount} rows affected")
        return cursor.rowcount
    except Exception as e:
        connection.rollback()
        print(f"The error '{e}' occurred")
        return None
    finally:
        cursor.close()

if __name__ == "__main__":
    # Load environment variables
    DB_NAME     = os.getenv("Postgres_db")      # e.g., TestDB
    DB_USER     = os.getenv("Postgres_user") or "acro0"
    DB_PASSWORD = os.getenv("Postgres_passwd")
    DB_HOST     = 'localhost'    # default to localhost
    DB_PORT     = int(os.getenv("Postgres_port") or 5432)

    conn = create_connection(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
    if not conn:
        exit(1)

    # Ensure we're in the correct database
    with conn.cursor() as cur:
        cur.execute("SELECT current_database();")
        print("Connected to database:", cur.fetchone()[0])

    # Create the employees table if it doesn't exist
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS public.employees (
        id SERIAL PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hire_date DATE NOT NULL
    );
    """
    execute_query(conn, create_table_sql)

    # Insert sample rows
    insert_sql = """
    INSERT INTO public.employees (first_name, last_name, email, hire_date) VALUES
      ('Jane', 'Doe', 'jane.doe@example.com', '2023-01-15'),
      ('John', 'Smith', 'john.smith@example.com', '2023-02-20'),
      ('Emily', 'Jones', 'emily.jones@example.com', '2023-03-10'),
      ('Michael', 'Brown', 'michael.brown@example.com', '2023-04-05'),
      ('Jessica', 'Williams', 'jessica.williams@example.com', '2023-05-18'),
      ('Chris', 'Davis', 'chris.davis@example.com', '2023-06-22'),
      ('Sarah', 'Miller', 'sarah.miller@example.com', '2023-07-30'),
      ('David', 'Wilson', 'david.wilson@example.com', '2023-08-01'),
      ('Laura', 'Moore', 'laura.moore@example.com', '2023-09-08'),
      ('Kevin', 'Taylor', 'kevin.taylor@example.com', '2023-10-14')
    ON CONFLICT (email) DO NOTHING;
    """
    affected = execute_query(conn, insert_sql)
    if affected is not None:
        print(f"Inserted up to {affected} new rows into employees table.")
    else:
        print("Insert operation failed.")

    conn.close()
    print("PostgreSQL connection closed")
