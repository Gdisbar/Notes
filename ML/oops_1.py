## ==============================================================================
## Inheritance
## ==============================================================================
class Person:
    def __init__(self, fname, lname):
        self.fname = fname
        self.lname = lname
class Employee(Person):
    all_employees = []
    def __init__(self, fname, lname, empid):
        Person.__init__(self, fname, lname)
        self.empid = empid
        Employee.all_employees.append(self)


# p1 = Person('George', 'smith')
# print(p1.fname, '-', p1.lname)  ## George - smith
# e1 = Employee('Jack', 'simmons', 456342)
# e2 = Employee('John', 'williams', 123656)
# print(e1.fname, '-', e1.empid)  ## Jack - 456342
# print(e2.fname, '-', e2.empid)  ## John - 123656

## ------------------------------------------------------------------------------
## Extending Built-in types
## ------------------------------------------------------------------------------
class EmployeesList(list):
    def search(self, name):
        matching_employees = []
        for employee in Employee.all_employees:
            if name in employee.fname:
                matching_employees.append(employee.fname)
        return matching_employees

class Employee(Person):
    all_employees = EmployeesList()
    def __init__(self, fname, lname, empid):
        Person.__init__(self, fname, lname)
        self.empid = empid
        Employee.all_employees.append(self)

# e1 = Employee('Jack', 'simmons', 456342)
# e2 = Employee('George', 'Brown', 656721)
# print(Employee.all_employees.search('or'))  ## ['George']

## ==============================================================================
## Polymorphism
## ==============================================================================
class Employee(Person):
    all_employees = EmployeesList ()
    def __init__(self, fname, lname, empid):
        Person.__init__(self, fname, lname)
        self.empid = empid
        Employee.all_employees.append(self)
    def getSalary(self):
        return 'You get Monthly salary.'
    def getBonus(self):
        return 'You are eligible for Bonus.'


class ContractEmployee(Employee):
   def getSalary(self):
        return 'You will not get Salary from Organization.'
   def getBonus(self):
        return 'You are not eligible for Bonus.'


# e1 = Employee('Jack', 'simmons', 456342)
# e2 = ContractEmployee('John', 'williams', 123656)
# print(e1.getBonus()) ## You are eligible for Bonus.
# print(e2.getBonus()) ## You are not eligible for Bonus.

## ==============================================================================
## Abstraction and Encapsulation
## ==============================================================================
# Abstraction means working with something you know how to use without 
# knowing how it works internally.

# Encapsulation allows binding data and associated methods together in a 
# unit i.e class.

# These principles together allows a programmer to define an interface for 
# applications, i.e. to define all tasks the program is capable to execute and 
# their respective input and output data.

# Example - We only need to know how TV Remote works in order to operate TV

# no underscores is a public one.
# a single underscore is private, however, still accessible from outside.
# double underscores is strongly private and not accessible from outside.

## ------------------------------------------------------------------------------
## a single underscore is private, however, still accessible from outside.
## ------------------------------------------------------------------------------

class Employee(Person):
    all_employees = EmployeesList()
    def __init__(self, fname, lname, empid):
        Person.__init__(self, fname, lname)
        self.__empid = empid
        Employee.all_employees.append(self)
    def getEmpid(self):
        return self.__empid


# e1 = Employee('Jack', 'simmons', 456342)
# print(e1.fname, e1.lname)  ## Jack simmons
# print(e1.getEmpid()) ## 456342
# print(e1._Employee__empid)  ##456342, Accessing through name mangling
# print(e1.__empid) ## AttributeError: 'Employee' object has no attribute '__empid'


## ------------------------------------------------------------------------------
## double underscores is strongly private and not accessible from outside.
## ------------------------------------------------------------------------------
class SecretAgentList:
    def __init__(self):
        self.agents = []

    def append(self, agent):
        self.agents.append(agent)


class Employee(Person):
    all_employees = EmployeesList()
    def __init__(self, fname, lname, empid):
        super().__init__(fname, lname)
        self.__empid = empid
        Employee.all_employees.append(self)
    def getEmpid(self):
        return self.__empid
    def __getEmpid(self):
    	raise AttributeError("Raise an Error.")
    	# return self.__empid

# e1 = Employee('Jack', 'simmons', 456342)
# print(e1.fname, e1.lname)  ## Jack simmons
# # print(e1.getEmpid()) ## 456342
# # print(e1._Employee__empid)  ##456342, Accessing through name mangling
# ## AttributeError: Raise an Error.but we can return 456342
# print(e1._Employee__getEmpid()) 
# print(e1.__empid) ## AttributeError: 'Employee' object has no attribute '__empid'


## ==============================================================================
## Exception
## ==============================================================================
class CustomError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

# try:
#     a = 2; b = 'hello'
#     if not (isinstance(a, int) and isinstance(b, int)):
#         raise CustomError('Two inputs must be integers.')
#     c = a**b
# except CustomError as e:
#     print(e) ## Two inputs must be integers.
# finally:
#     print("In finally clause.") ## In finally clause.

## ------------------------------------------------------------------------------
## else clause is also an optional clause with try ... except
## ------------------------------------------------------------------------------

try:
    a = 14 / 7
except ZeroDivisionError:
    print('oops!!!')
else:
    print('First ELSE') ## First ELSE
try:
    a = 14 / 0
except ZeroDivisionError:
    print('oops!!!')       ## oops!!!
else:
    print('Second ELSE')

## ==============================================================================
## Function & Closure
## ==============================================================================

## ------------------------------------------------------------------------------
## It can be functioned as a data and be assigned to a variable.
## ------------------------------------------------------------------------------
def greet():
    return 'Hello Everyone!'

# print(greet())      ## Hello Everyone!
# wish = greet        # 'greet' function assigned to variable 'wish'
# print(type(wish))   ## <class 'function'>
# print(wish())       ## Hello Everyone! 

## ------------------------------------------------------------------------------
## It can accept any other function as an argument.
## ------------------------------------------------------------------------------

def add(x, y):
    return x + y
def sub(x, y):
   return x - y
def prod(x, y):
    return x * y
def do(func, x, y):
   return func(x, y)

# print(do(add, 12, 4))  # 'add' as arg
# print(do(sub, 12, 4))  # 'sub' as arg
# print(do(prod, 12, 4))  # 'prod' as arg

## ------------------------------------------------------------------------------
## It can return a function as its result.
## ------------------------------------------------------------------------------

def outer():
    def inner():
        s = 'Hello world!'
        return s            
    return inner  # add '()' to return output of 'inner' function


# print(outer()) ## returns 'inner' function itself : <function outer.<locals>.inner at 0x7f1c5a615620>
# func = outer() 
# print(type(func)) ## <class 'function'>
# print(func()) ## Hello world! : calling 'inner' function

## ------------------------------------------------------------------------------
## Closures
## ------------------------------------------------------------------------------

def multiple_of(x):
    def multiple(y):
        return x*y
    return multiple

c1 = multiple_of(5)  # 'c1' is a closure
c2 = multiple_of(6)  # 'c2' is a closure
# print(c1(4)) ## 20
# print(c2(4)) ## 24