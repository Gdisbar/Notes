## ==============================================================================
## Decorators
## ==============================================================================

## ------------------------------------------------------------------------------
## creation of closure function wish using the higher order function outer
## ------------------------------------------------------------------------------

def outer(func):
    def inner():
        print("Accessing :",func.__name__)
        return func()
    return inner

def greet():
   print('Hello!')

wish = outer(greet)
# wish()

## Accessing : greet
## Hello!



## ------------------------------------------------------------------------------
## creation of decorator function outer, which is used to decorate function greet. 
## ------------------------------------------------------------------------------

def outer(func):
	def inner():
	    print("Accessing :",func.__name__)
	    return func()
	return inner

def greet():
   return 'Hello!' ## Here we're returning a value

wish = outer(greet) # decorating 'greet'
# wish()  ## Accessing : greet



## ------------------------------------------------------------------------------
## decorating the greet function with decorator function, outer, using @ symbol.
## ------------------------------------------------------------------------------

def outer(func):
    def inner():
        print("Accessing :", func.__name__)
        return func()
    return inner

@outer
def greet():
    return 'Hello!'

# greet() ## Accessing : greet


## ==============================================================================
## getter & setter method 
## ==============================================================================

## ------------------------------------------------------------------------------
## Descriptors
## ------------------------------------------------------------------------------


class EmpNameDescriptor:
    def __get__(self, obj, owner):
        return self.__empname
    def __set__(self, obj, value):
        if not isinstance(value, str):
            raise TypeError("'empname' must be a string.")
        self.__empname = value

class EmpIdDescriptor:
    def __get__(self, obj, owner):
        return self.__empid
    def __set__(self, obj, value):
        if hasattr(obj, 'empid'):
            raise ValueError("'empid' is read only attribute")
        if not isinstance(value, int):
            raise TypeError("'empid' must be an integer.")
        self.__empid = value


class Employee:
    empid = EmpIdDescriptor()           
    empname = EmpNameDescriptor()       
    def __init__(self, emp_id, emp_name):
        self.empid = emp_id
        self.empname = emp_name


# e1 = Employee(123456, 'John')
# print(e1.empid, '-', e1.empname)  ## 123456 - John
# e1.empname = 'Williams'
# print(e1.empid, '-', e1.empname) ## 123456 - Williams
# e1.empid = 76347322  # ValueError: 'empid' is read only attribute

## ------------------------------------------------------------------------------
## Properties - property(fget=None, fset=None, fdel=None, doc=None)
## ------------------------------------------------------------------------------


class Employee:
    def __init__(self, emp_id, emp_name):
        self.__empid = emp_id  # Use double underscores for private attributes
        self.__empname = emp_name

    def getEmpID(self):
        return self.__empid

    def setEmpID(self, value):
        if not isinstance(value, int):
            raise TypeError("'empid' must be an integer.")
        self.__empid = value

    empid = property(getEmpID, setEmpID)

    def getEmpName(self):
        return self.__empname

    def setEmpName(self, value):
        if not isinstance(value, str):
            raise TypeError("'empname' must be a string.")
        self.__empname = value

    def delEmpName(self):
        del self.__empname

    empname = property(getEmpName, setEmpName, delEmpName)


# e1 = Employee(123456, 'John')
# print(e1.empid, '-', e1.empname)  ## '123456 - John'
# del e1.empname  # Deletes 'empname'
# print(e1.empname)  
## AttributeError: 'Employee' object has no attribute '_Employee__empname'. 
## Did you mean: '_Employee__empid'?

## ------------------------------------------------------------------------------
## Property - Using Decorator
## ------------------------------------------------------------------------------


class Employee:
    def __init__(self, emp_id, emp_name):
        self.empid = emp_id
        self.empname = emp_name

    @property
    def empid(self):
        return self.__empid
    @empid.setter
    def empid(self, value):
        if not isinstance(value, int):
            raise TypeError("'empid' must be an integer.")
        self.__empid = value

    @property
    def empname(self):
        return self.__empname
    @empname.setter
    def empname(self, value):
        if not isinstance(value, str):
            raise TypeError("'empname' must be a string.")
        self.__empname = value
    @empname.deleter
    def empname(self):
        del self.__empname


# e1 = Employee(123456, 'John')
# print(e1.empid, '-', e1.empname)    # -> '123456 - John'
# del e1.empname    # Deletes 'empname'
# print(e1.empname) 
# ## AttributeError: 'Employee' object has no attribute '_Employee__empname'. 
# ## Did you mean: '_Employee__empid'?

## ==============================================================================
## Class and Static Methods 
## ==============================================================================

## ------------------------------------------------------------------------------
## Class method - no object information : TypeError
## ------------------------------------------------------------------------------


class Circle(object):
    no_of_circles = 0
    def __init__(self, radius):
        self.__radius = radius
        Circle.no_of_circles += 1
    def getCirclesCount(self):
        return Circle.no_of_circles

c1 = Circle(3.5)
c2 = Circle(5.2)
# c3 = Circle(4.8)

# print(c1.getCirclesCount())     ## 2, uncomment c3 will get 3
# print(c2.getCirclesCount())     ## 2 . uncomment c3 will get 3
# # print(Circle.getCirclesCount(c3)) ## 3
# print(Circle.getCirclesCount()) 
# ## TypeError: Circle.getCirclesCount() missing 1 required positional 
# ## argument: 'self' ---> Circle without any object information



## ------------------------------------------------------------------------------
## Class method - with object information
## ------------------------------------------------------------------------------

class Circle(object):
    no_of_circles = 0
    def __init__(self, radius):
        self.__radius = radius
        Circle.no_of_circles += 1

    @classmethod
    def getCirclesCount(self):
        return Circle.no_of_circles

c1 = Circle(3.5)
c2 = Circle(5.2)
c3 = Circle(4.8)

# print(c1.getCirclesCount())     # -> 3
# print(c2.getCirclesCount())     # -> 3
# print(Circle.getCirclesCount()) # -> 3


## ------------------------------------------------------------------------------
## Static method
## ------------------------------------------------------------------------------

# def square(x):
#         return x**2

class Circle(object):
    def __init__(self, radius):
        self.__radius = radius
    def area(self):
        return 3.14*square(self.__radius)

# c1 = Circle(3.9)
# print(c1.area())  ## 47.7594
# print(square(10)) ## 100

## ------------------------------------------------------------------------------
## Static method - using decorator
## ------------------------------------------------------------------------------


class Circle(object):
    def __init__(self, radius):
        self.__radius = radius
    
    @staticmethod
    def square(x):
        return x**2

    def area(self):
        return 3.14*self.square(self.__radius)

# c1 = Circle(3.9)
# print(c1.area())  
# print(Circle.square(10)) # -> 100
# print(c1.square(10))     # -> 100
# print(square(10))  ## NameError: name 'square' is not defined


## ==============================================================================
## Abstract Class
## ==============================================================================

from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    @abstractmethod
    def perimeter(self):
        pass

# s1 = Shape() 
# TypeError: Can't instantiate abstract class Shape with abstract 
# methods area, perimeter

## ------------------------------------------------------------------------------
## Need to override all abstract methods , else : TypeError
## ------------------------------------------------------------------------------

class Circle(Shape):
    def __init__(self, radius):
        self.__radius = radius
    @staticmethod
    def square(x):
        return x**2
    def area(self):
        return 3.14*self.square(self.__radius)

# c1 = Circle(3.9)
## TypeError: Can't instantiate abstract class 
## Circle with abstract method perimeter

## ------------------------------------------------------------------------------
## After overriding all abstract methods 
## ------------------------------------------------------------------------------

class Circle(Shape):
    def __init__(self, radius):
        self.__radius = radius
    @staticmethod
    def square(x):
        return x**2
    def area(self):
        return 3.14*self.square(self.__radius)
    def perimeter(self):
        return 2*3.14*self.__radius

c1 = Circle(3.9)
# print(c1.area()) ## 47.7594

## ==============================================================================
## Context Manager
## ==============================================================================

# import sqlite3
# try:
#     dbConnection = sqlite3.connect('TEST.db')
#     cursor = dbConnection.cursor()
#     '''
#     Few db operations
#     ...
#     '''
# except Exception:
#     print('No Connection.')
# finally:
#     dbConnection.close()

# ## ------------------------------------------------------------------------------

# import sqlite3
# class DbConnect(object):
#     def __init__(self, dbname):
#         self.dbname = dbname
#     def __enter__(self):
#         self.dbConnection = sqlite3.connect(self.dbname)
#         return self.dbConnection
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.dbConnection.close()
# with DbConnect('TEST.db') as db:
#     cursor = db.cursor()
#     '''
#    Few db operations
#    ...
#     '''

## ==============================================================================
## Coroutines
## ==============================================================================

# A Coroutine is generator which is capable of constantly receiving input data, 
# process input data and may or may not return any output.

# Execution of a coroutine stops when it reaches yield statement.

# It uses send method to send any input value, which is captured by yield expression.

def TokenIssuer():
    tokenId = 0
    while True:
        name = yield
        tokenId += 1
        print('Token number of', name, ':', tokenId)

# t = TokenIssuer()
# next(t)
# t.send('George') ## Token number of George : 1
# t.send('Rosy')   ## Token number of Rosy : 2
# t.send('Smith')  ## Token number of Smith : 3

## ------------------------------------------------------------------------------
## Using custom start value
## ------------------------------------------------------------------------------


def TokenIssuer(tokenId=0):
    try:
       while True:
            name = yield
            tokenId += 1
            print('Token number of', name, ':', tokenId)
    except GeneratorExit:
        print('Last issued Token is :', tokenId)

# t = TokenIssuer(100)
# next(t)
# t.send('George')
# t.send('Rosy')
# t.send('Smith')
# t.close()

# # Token number of George : 101
# # Token number of Rosy : 102
# # Token number of Smith : 103
# # Last issued Token is : 103

## ------------------------------------------------------------------------------
## Using custom start value + coroutine decorator
## ------------------------------------------------------------------------------

def coroutine_decorator(func):
    def wrapper(*args, **kwdargs):
        c = func(*args, **kwdargs)
        next(c)
        return c
    return wrapper


@coroutine_decorator
def TokenIssuer(tokenId=0):
    try:
        while True:
            name = yield
            tokenId += 1
            print('Token number of', name, ':', tokenId)
    except GeneratorExit:
        print('Last issued Token is :', tokenId)

# t = TokenIssuer(10)
# t.send('George')
# t.send('Rosy')
# t.send('Smith')
# t.close()

# # Token number of George : 11
# # Token number of Rosy : 12
# # Token number of Smith : 13
# # Last issued Token is : 13