from typing import Optional, List # Supports for type hints
from pydantic import BaseModel # Most widely used data validation library for python
from enum import Enum # Supports for enumerations

# enum for gender
class Gender(str, Enum):
    male = "male"
    female = "female"

# enum for role
class Role(str, Enum):
    admin = "admin"
    user = "user"
    student = "student"
    teacher = "teacher"

class User(BaseModel):
    first_name: str
    last_name: str
    middle_name: Optional[str] = None  # Make middle name optional
    gender: Gender
    email_address: str
    phone_number: str
    roles: List[Role] # user can have several roles



class UpdateUserDTO(BaseModel):
    first_name: Optional[str] = None 
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    gender: Optional[str] = None 
    phone_number: Optional[str] = None 
    roles: Optional[list[str]] = None 

    class Config:
        extra = "ignore"  # Ignore extra fields in the request body