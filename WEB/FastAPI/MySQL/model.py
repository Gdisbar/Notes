from typing import Optional, List
from pydantic import BaseModel, EmailStr
from enum import Enum

# Enum for gender
class Gender(str, Enum):
    male = "male"
    female = "female"

# Enum for role
class Role(str, Enum):
    admin = "admin"
    user = "user"
    student = "student"
    teacher = "teacher"

class User(BaseModel):
    first_name: str
    last_name: str
    middle_name: Optional[str] = None
    gender: Gender
    email_address: EmailStr
    phone_number: str
    roles: Optional[List[Role]] = None

class UpdateUserDTO(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    gender: Optional[Gender] = None
    phone_number: Optional[str] = None
    roles: Optional[List[Role]] = None

    class Config:
        extra = "ignore"  # Ignore extra fields in the request body
