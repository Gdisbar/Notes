from fastapi import FastAPI, HTTPException
from pydantic import EmailStr
from typing import Optional, List
import mysql.connector
from dotenv import load_dotenv
import os
import asyncio
from model import Gender, Role, User, UpdateUserDTO
from mysql.connector.errors import IntegrityError

load_dotenv()

app = FastAPI()

# Async database connection
async def connect():
    return await asyncio.to_thread(
        mysql.connector.connect,
        host=os.environ.get('DB_HOST'),
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASSWORD'),
        database=os.environ.get('DB_NAME')
    )

async def execute_query(cursor, query, params=None):
    return await asyncio.to_thread(cursor.execute, query, params)

async def initialize_db():
    """
    Initializes the database and creates tables if they don't exist.
    """
    conn = await connect()
    try:
        cursor = conn.cursor()
        await execute_query(cursor, """
            CREATE TABLE IF NOT EXISTS gender (
                value VARCHAR(10) PRIMARY KEY,
                CHECK (value IN ('male', 'female'))
            )
        """)
        await execute_query(cursor, """
            INSERT IGNORE INTO gender (value) VALUES ('male'), ('female')
        """)

        await execute_query(cursor, """
            CREATE TABLE IF NOT EXISTS role (
                value VARCHAR(20) PRIMARY KEY,
                CHECK (value IN ('admin', 'user', 'student', 'teacher'))
            )
        """)
        await execute_query(cursor, """
            INSERT IGNORE INTO role (value) VALUES ('admin'), ('user'), ('student'), ('teacher')
        """)

        await execute_query(cursor, """
            CREATE TABLE IF NOT EXISTS users (
                first_name VARCHAR(20),
                last_name VARCHAR(20),
                middle_name VARCHAR(20) NULL,
                gender ENUM('male', 'female'),
                email_address VARCHAR(50) PRIMARY KEY,
                phone_number VARCHAR(15),
                roles TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()

@app.on_event("startup")
async def startup_event():
    await initialize_db()
# C <= Create
@app.post("/api/v1/create-user", response_model=User)
async def create_user(user_data: User):
    conn = await connect()
    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO users (first_name, last_name, middle_name, gender, email_address, phone_number, roles)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        roles_str = ",".join(user_data.roles) if user_data.roles else None
        params = (
            user_data.first_name,
            user_data.last_name,
            user_data.middle_name,
            user_data.gender.value,
            user_data.email_address,
            user_data.phone_number,
            roles_str,
        )
        await execute_query(cursor, query, params)
        conn.commit()
        return user_data
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Email already exists")
    finally:
        conn.close()

# R <=== Read
# Read all users
@app.get("/api/v1/read-all-users", response_model=List[User])
async def read_users():
    conn = await connect()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        users = []
        for row in rows:
            user_roles = row["roles"].split(",") if row["roles"] else []
            users.append(User(**{**row, "roles": user_roles}))
        return users
    finally:
        conn.close()

# Read one user by email_address
@app.get("/api/v1/read-user/{email_address}", response_model=User)
async def read_user_by_email(email_address: str):
    conn = await connect()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email_address = %s", (email_address,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        user_roles = row["roles"].split(",") if row["roles"] else []
        return User(**{**row, "roles": user_roles})
    finally:
        conn.close()

# U <=== Update
@app.put("/api/v1/update-user/{email_address}", response_model=User)
async def update_user(email_address: str, user_data: UpdateUserDTO):
    conn = await connect()
    try:
        cursor = conn.cursor(dictionary=True)
        update_fields = []
        params = []

        for key, value in user_data.dict(exclude_unset=True).items():
            if key == "roles":
                value = ",".join(value) if value else None
            update_fields.append(f"{key} = %s")
            params.append(value)

        if not update_fields:
            raise HTTPException(status_code=400, detail="No data provided to update")

        params.append(email_address)
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE email_address = %s"
        await execute_query(cursor, query, params)
        conn.commit()
        return await read_user_by_email(email_address)
    finally:
        conn.close()

# D <=== Delete
# Delete user by email_address
@app.delete("/api/v1/delete-user/{email_address}")
async def delete_user(email_address: str):
    conn = await connect()
    try:
        cursor = conn.cursor()
        query = "DELETE FROM users WHERE email_address = %s"
        await execute_query(cursor, query, (email_address,))
        conn.commit()
        return {"message": "User deleted successfully"}
    finally:
        conn.close()
