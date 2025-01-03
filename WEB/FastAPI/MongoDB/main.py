# import fast api
from fastapi import FastAPI, HTTPException
from models import Gender,Role,User,UpdateUserDTO # import the user model defined by us
# imports for the MongoDB database connection
from motor.motor_asyncio import AsyncIOMotorClient
# import for fast api lifespan
from contextlib import asynccontextmanager
from typing import List # Supports for type hints
from pydantic import BaseModel # Most widely used data validation library for python
import json  # For serializing/deserializing user data
from redis.asyncio import Redis

# define a lifespan method for fastapi
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the database connection
    await startup_db_client(app)
    yield
    # Close the database connection
    await shutdown_db_client(app)


# method for start the MongoDb Connection
async def startup_db_client(app):
    app.mongodb_client = AsyncIOMotorClient("mongodb://localhost:27017/")
    app.mongodb = app.mongodb_client.get_database("college")
    app.redis_client = Redis(host="localhost", port=6379, decode_responses=True)
    print("MongoDB & Redis connected.")

# method to close the database connection
async def shutdown_db_client(app):
    app.mongodb_client.close()
    await app.redis_client.close()
    print("Database & Redis disconnected.")

# creating a server with python FastAPI
app = FastAPI(lifespan=lifespan)

# C <=== Create
@app.post("/api/v1/create-user", response_model=User)
async def insert_user(user_data: User):
    result = await app.mongodb["users"].insert_one(user_data.dict())
    inserted_user = await app.mongodb["users"].find_one({"_id": result.inserted_id})
    # Invalidate Redis cache
    await app.redis_client.delete("read-all-users")
    return inserted_user

# R <=== Read
# Read all users
@app.get("/api/v1/read-all-users", response_model=List[User])
async def read_users():
    redis_key = "read-all-users"
    cached_data = await app.redis_client.get(redis_key)
    if cached_data:
        print("Returning cached results")
        return json.loads(cached_data)

    users = await app.mongodb["users"].find().to_list(None)
    users_serialized = [User(**user).dict(by_alias=True) for user in users]
    # Cache the results in Redis for 10 minutes
    await app.redis_client.set(redis_key, json.dumps(users_serialized), ex=600)
    return users_serialized

# Read one user by email_address
@app.get("/api/v1/read-user/{email_address}", response_model=User)
async def read_user_by_email(email_address: str):
    user = await app.mongodb["users"].find_one({"email_address": email_address})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# U <=== Update
@app.put("/api/v1/update-user/{email_address}", response_model=User)
async def update_user(email_address: str, user_data: UpdateUserDTO):
    """
    Updates a user with the given email_address.
    """
    # Retrieve the existing user document
    existing_user = await app.mongodb["users"].find_one({"email_address": email_address})
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Create a dictionary of updates from the provided data - both are same
    update_data = user_data.dict(exclude_unset=True) 
    # update_data = {k: v for k, v in user_update.dict().items() if v is not None}  # Filter out None values

    # Update the existing user document with the changes
    updated_result = await app.mongodb["users"].update_one(
        {"email_address": email_address}, 
        {"$set": update_data}
    )

    if updated_result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found or no update needed")

    # Retrieve the updated user document
    updated_user = await app.mongodb["users"].find_one({"email_address": email_address})
    # Invalidate Redis cache
    await app.redis_client.delete("read-all-users")
    return updated_user

# D <=== Delete
# Delete user by email_address
@app.delete("/api/v1/delete-user/{email_address}", response_model=dict)
async def delete_user_by_email(email_address: str):
    delete_result = await app.mongodb["users"].delete_one({"email_address": email_address})
    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    # Invalidate Redis cache
    await app.redis_client.delete("read-all-users")
    return {"message": "User deleted successfully"}
