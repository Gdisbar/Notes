from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os 

load_dotenv()

username = os.getenv("Mongodb_user")
passwd = os.getenv("Mongodb_passwd")

# Connection URI (replace placeholders)
uri = f"mongodb://{username}:{passwd}@localhost:27017/?authSource=admin"
client = MongoClient(uri)

# Reference your new database and collection
db = client["myNewDatabase"]
collection = db["myCollection"]

# Insert a document to create the database and collection
result = collection.insert_one({
    "name": "First Document",
    "createdAt": datetime.now()
})

print("Inserted document ID:", result.inserted_id)
