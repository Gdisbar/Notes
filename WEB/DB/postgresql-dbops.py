import os
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

combined_text_list = [
    "What is the weather today? It is sunny and 25°C.",
    "Tell me a joke. Why don’t skeletons fight each other? They don’t have the guts!",
    "How do I reset my password? Click on 'Forgot Password' and follow the instructions.",
    "What is 2 + 2? 2 + 2 equals 4.",
    "End the session. Session has been ended. Thank you!"
]

# Database connection parameters
db_params = {
    'host': 'localhost',
    'dbname': os.environ["DBNAME"],
    'user': os.environ["USERNAME"],
    'password': os.environ["PASSWORD"],
    'port': '5432'
}

# Load data and embeddings from the CSV file
embeddings = pd.read_csv('embeddings.csv', header=None)  # Ensure no header issues
print(f"Number of combined_text entries: {len(combined_text_list)}")
print(f"Number of embeddings: {embeddings.shape[0]}")

try:
    # Establish connection to PostgreSQL
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Insert the data into the vector database table
    for index in range(len(combined_text_list)):
        chat_id = index
        combined_query_response = combined_text_list[index]
        vector_embedding = embeddings.iloc[index].to_list()  # Convert to list

        # Insert data into the table
        cursor.execute("""
            INSERT INTO ChatDB.ChatData_Vector (chat_id, combined_query_response, vector_embedding)
            VALUES (%s, %s, %s);
        """, (chat_id, combined_query_response, vector_embedding))
        print(f"Inserting data for index {index}: {combined_query_response}")

    print("Data successfully uploaded to PostgreSQL vector database.")

    # Commit and close the connection
    conn.commit()
    cursor.close()
    conn.close()

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL:", error)
