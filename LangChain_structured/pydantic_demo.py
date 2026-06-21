from pydantic import BaseModel,EmailStr,Field
from typing import Optional

# pydantic can do emplicit type conversion..
# Build in validation for many things like email.
class Student(BaseModel):
    name:str = 'samee'
    age:Optional[int]=None
    email: EmailStr
    cgpa:float=Field(gt=0,it=0)

new_student={'name':'nitish','email':'abc','cgpa':5}

student=Student(**new_student)
# can interchange between different formats like dict or json string libarires
print(student)