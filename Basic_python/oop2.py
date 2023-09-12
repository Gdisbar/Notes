
================|
Polymorphism    |
================|

# Python program to demonstrate in-built poly- 
# morphic functions 

# len() being used for a string 
print(len("geeks")) 

# len() being used for a list 
print(len([10, 20, 30])) 


class India(): 
	def capital(self): 
		print("New Delhi is the capital of India.") 

	def language(self): 
		print("Hindi is the most widely spoken language of India.") 

	def type(self): 
		print("India is a developing country.") 

class USA(): 
	def capital(self): 
		print("Washington, D.C. is the capital of USA.") 

	def language(self): 
		print("English is the primary language of USA.") 

	def type(self): 
		print("USA is a developed country.") 

obj_ind = India() 
obj_usa = USA() 
for country in (obj_ind, obj_usa): 
	country.capital() 
	country.language() 
	country.type() 


Output:

		New Delhi is the capital of India.
		Hindi is the most widely spoken language of India.
		India is a developing country.
		Washington, D.C. is the capital of USA.
		English is the primary language of USA.
		USA is a developed country.


Polymorphism with Inheritance:
--------------------------------
In Python, Polymorphism lets us define methods in the child class that have the same name as 
the methods in the parent class. In inheritance, the child class inherits the methods from the parent class. 
However, it is possible to modify a method in a child class that it has inherited from the parent class. 
This is particularly useful in cases where the method inherited from the parent class doesn’t quite fit the child class. 
In such cases, we re-implement the method in the child class. 
This process of re-implementing a method in the child class is known as Method Overriding.

class Bird: 
def intro(self): 
	print("There are many types of birds.") 
	
def flight(self): 
	print("Most of the birds can fly but some cannot.") 
	
class sparrow(Bird): 
def flight(self): 
	print("Sparrows can fly.") 
	
class ostrich(Bird): 
def flight(self): 
	print("Ostriches cannot fly.") 
	
obj_bird = Bird() 
obj_spr = sparrow() 
obj_ost = ostrich() 

obj_bird.intro() 
obj_bird.flight() 

obj_spr.intro() 
obj_spr.flight() 

obj_ost.intro() 
obj_ost.flight() 

Output:

		There are many types of birds.
		Most of the birds can fly but some cannot.
		There are many types of birds.
		Sparrows can fly.
		There are many types of birds.
		Ostriches cannot fly.


Polymorphism with a Function and objects:
It is also possible to create a function that can take any object, allowing for polymorphism. 
In this example, let’s create a function called “func()” which will take an object which we will name “obj”. 
Though we are using the name ‘obj’, any instantiated object will be able to be called into this function. 
Next, lets give the function something to do that uses the ‘obj’ object we passed to it.
In this case lets call the three methods, viz., capital(), language() and type(), 
each of which is defined in the two classes ‘India’ and ‘USA’. Next, 
let’s create instantiations of both the ‘India’ and ‘USA’ classes if we don’t have them already. 
With those, we can call their action using the same func() function:


class India(): 
	def capital(self): 
		print("New Delhi is the capital of India.") 

	def language(self): 
		print("Hindi is the most widely spoken language of India.") 

	def type(self): 
		print("India is a developing country.") 

class USA(): 
	def capital(self): 
		print("Washington, D.C. is the capital of USA.") 

	def language(self): 
		print("English is the primary language of USA.") 

	def type(self): 
		print("USA is a developed country.") 

def func(obj): 
	obj.capital() 
	obj.language() 
	obj.type() 

obj_ind = India() 
obj_usa = USA() 

func(obj_ind) 
func(obj_usa) 


Output:

		New Delhi is the capital of India.
		Hindi is the most widely spoken language of India.
		India is a developing country.
		Washington, D.C. is the capital of USA.
		English is the primary language of USA.
		USA is a developed country.


Class or static variables are shared by all objects. 
Instance or non-static variables are different for different objects (every object has a copy of it).

The Python approach is simple, it doesn’t require a static keyword. All variables which are assigned a 
value in class declaration are class variables. 
And variables which are assigned values inside methods are instance variables.
# Python program to show that the variables with a value 
# assigned in class declaration, are class variables 

# Class for Computer Science Student 
class CSStudent: 
	stream = 'cse'				 # Class Variable (static variable)
	def __init__(self,name,roll): 
		self.name = name		 # Instance Variable (non-static)
		self.roll = roll		 # Instance Variable 

# Objects of CSStudent class 
a = CSStudent('Geek', 1) 
b = CSStudent('Nerd', 2) 

print(a.stream) # prints "cse" 
print(b.stream) # prints "cse" 
print(a.name) # prints "Geek" 
print(b.name) # prints "Nerd" 
print(a.roll) # prints "1" 
print(b.roll) # prints "2" 

# Class variables can be accessed using class 
# name also 
print(CSStudent.stream) # prints "cse" 

Output:

		cse
		cse
		Geek
		Nerd
		1
		2
		cse


====================================|
class method & static method        |
====================================|
Class Method
-------------------
The @classmethod decorator, is a builtin function decorator that is an expression that gets evaluated after your 
function is defined. The result of that evaluation shadows your function definition.
A class method receives the class as implicit first argument, just like an instance method receives the instance

Syntax:

class C(object):
    @classmethod
    def fun(cls, arg1, arg2, ...):
       ....
fun: function that needs to be converted into a class method
returns: a class method for function.


It can modify a class state that would apply across all the instances of the class. 
For example it can modify a class variable that will be applicable to all the instances.

Static Method
-------------------
A static method does not receive an implicit first argument.
Syntax:

class C(object):
    @staticmethod
    def fun(arg1, arg2, ...):
        ...
returns: a static method for function fun.

A static method is also a method which is bound to the class and not the object of the class.
A static method can’t access or modify class state.


Class method vs Static Method
--------------------------------
A class method can access or modify class state while a static method can’t access or modify it.
In general, static methods know nothing about class state. 
They are utility type methods that take some parameters and work upon those parameters. 
On the other hand class methods must have class as parameter.
We use @classmethod decorator in python to create a class method and we use 
@staticmethod decorator to create a static method in python.

When to use what?

We generally use class method to create factory methods. 
Factory methods return class object ( similar to a constructor ) for different use cases.
We generally use static methods to create utility functions.

# Python program to demonstrate 
# use of class method and static method. 
from datetime import date 

class Person: 
	def __init__(self, name, age): 
		self.name = name 
		self.age = age 
	
	# a class method to create a Person object by birth year. 
	@classmethod
	def fromBirthYear(cls, name, year): 
		return cls(name, date.today().year - year) 
	
	# a static method to check if a Person is adult or not. 
	@staticmethod
	def isAdult(age): 
		return age > 18

person1 = Person('mayank', 21) 
person2 = Person.fromBirthYear('mayank', 1996) 

print (person1.age) 
print (person2.age )

# print the result 
print(Person.isAdult(22))

Output:

		21
		21
		True


We should be careful when changing value of class variable. If we try to change class variable using object, 
a new instance (or non-static) variable for 
that particular object is created and this variable shadows the class variables. 

# Class for Computer Science Student 
class CSStudent: 
	stream = 'cse'	 # Class Variable 
	def __init__(self, name, roll): 
		self.name = name 
		self.roll = roll 

# Driver program to test the functionality 
# Creating objects of CSStudent class 
a = CSStudent("Geek", 1) 
b = CSStudent("Nerd", 2) 

print ("Initially")
print("a.stream =", a.stream) 
print("b.stream =", b.stream)

# This thing doesn't change class(static) variable 
# Instead creates instance variable for the object 
# 'a' that shadows class member. 
a.stream = "ece"

print("\nAfter changing a.stream")
print("a.stream =", a.stream)
print("b.stream =", b.stream)

Output:

		Initially
		a.stream = cse
		b.stream = cse

		After changing a.stream
		a.stream = ece
		b.stream = cse

# Program to show how to make changes to the 
# class variable in Python 

# Class for Computer Science Student 
class CSStudent: 
	stream = 'cse'	 # Class Variable 
	def __init__(self, name, roll): 
		self.name = name 
		self.roll = roll 

# New object for further implementation 
a = CSStudent("check", 3) 
print("a.tream =", a.stream) 

# Correct way to change the value of class variable 
CSStudent.stream = "mec"
print("\nClass variable changes to mec")

# New object for further implementation 
b = CSStudent("carter", 4) 

print("\nValue of variable steam for each object")
print("a.stream =", a.stream) 
print("b.stream =", b.stream)

Output:

		a.tream = cse

		Class variable changes to mec

		Value of variable steam for each object
		a.stream = mec
		b.stream = mec

==========================|
Constructor/Destructor    |
==========================|

Default Constructor
-----------------------
class GeekforGeeks: 

	# default constructor 
	def __init__(self): 
		self.geek = "GeekforGeeks"

	# a method for printing data members 
	def print_Geek(self): 
		print(self.geek) 


# creating object of the class 
obj = GeekforGeeks() 

# calling the instance method using the object obj 
obj.print_Geek() 

Output :

		GeekforGeeks

Parameterized Constructor
-------------------------------
class Addition: 
	first = 0
	second = 0
	answer = 0
	
	# parameterized constructor 
	def __init__(self, f, s): 
		self.first = f 
		self.second = s 
	
	def display(self): 
		print("First number = " + str(self.first)) 
		print("Second number = " + str(self.second)) 
		print("Addition of two numbers = " + str(self.answer)) 

	def calculate(self): 
		self.answer = self.first + self.second 

# creating object of the class 
# this will invoke parameterized constructor 
obj = Addition(1000, 2000) 

# perform Addition 
obj.calculate() 

# display result 
obj.display() 

Output :

		First number = 1000
		Second number = 2000
		Addition of two numbers = 3000


# Python program to illustrate destructor 
class Employee: 

	# Initializing 
	def __init__(self): 
		print('Employee created.') 

	# Deleting (Calling destructor) 
	def __del__(self): 
		print('Destructor called, Employee deleted.') 

obj = Employee() 
del obj 

Output:

		Employee created.
		Destructor called, Employee deleted.



# Python program to illustrate destructor 

class Employee: 

	# Initializing 
	def __init__(self): 
		print('Employee created') 

	# Calling destructor 
	def __del__(self): 
		print("Destructor called") 

def Create_obj(): 
	print('Making Object...') 
	obj = Employee() 
	print('function end...') 
	return obj 

print('Calling Create_obj() function...') 
obj = Create_obj() 
print('Program End...') 


Output:

		Calling Create_obj() function...
		Making Object...
		Employee created
		function end...
		Program End...
		Destructor called


# Python program to illustrate destructor 

class A: 
	def __init__(self, bb): 
		self.b = bb 

class B: 
	def __init__(self): 
		self.a = A(self) 
	def __del__(self): 
		print("die") 

def fun(): 
	b = B() 

fun() 

Output:

		die





