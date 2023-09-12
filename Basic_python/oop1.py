===================|
Inheritance        |
===================|
class derived-classname(superclass-name) 

class Employee:

    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)


class Developer(Employee):
    raise_amt = 1.10

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        self.prog_lang = prog_lang


class Manager(Employee):

    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('-->', emp.fullname())


dev_1 = Developer('Corey', 'Schafer', 50000, 'Python')
dev_2 = Developer('Test', 'Employee', 60000, 'Java')

mgr_1 = Manager('Sue', 'Smith', 90000, [dev_1])

print(mgr_1.email) #Sue.Smith@email.com

mgr_1.add_emp(dev_2)
mgr_1.remove_emp(dev_2)

mgr_1.print_emps() # --> Corey Schafer

-----------------------------------------------------------------------------------
# A Python program to demonstrate working of inheritance 
class Pet: 
        #__init__ is an constructor in Python 
        def __init__(self, name, age):   
                self.name = name 
                self.age = age 

# Class Cat inheriting from the class Pet 
class Cat(Pet):      
        def __init__(self, name, age): 
                # calling the super-class function __init__ 
                # using the super() function 
                super().__init__(name, age) 

def Main(): 
        thePet = Pet("Pet", 1) 
        jess = Cat("Jess", 3) 
        
        # isinstance() function to check whether a class is 
        # inherited from another class 
        print("Is jess a cat? " +str(isinstance(jess, Cat)))          # True
        print("Is jess a pet? " +str(isinstance(jess, Pet)))          # True
        print("Is the pet a cat? "+str(isinstance(thePet, Cat)))      # False
        print("Is thePet a Pet? " +str(isinstance(thePet, Pet)))      # True
        print(jess.name)                                              # Jess

if __name__=='__main__': 
        Main() 

--------------------------------------------------------------------------------

# Python program to show that we can create instance variables inside methods 

class CSStudent: 
    
    # Class Variable 
    stream = 'cse'  
    
    # The init method or constructor 
    def __init__(self, roll): 
        
        # Instance Variable 
        self.roll = roll             

    # Adds an instance variable 
    def setAddress(self, address): 
        self.address = address 
    
    # Retrieves instance variable    
    def getAddress(self):    
        return self.address 

# Driver Code 
a = CSStudent(101) 
a.setAddress("Noida, UP") 
print(a.getAddress())  # Noida, UP

--------------------------------------------------------------------------------

# A Python program to demonstrate inheritance 

# Base or Super class. Note object in bracket. 
# (Generally, object is made ancestor of all classes) 
# In Python 3.x "class Person" is 
# equivalent to "class Person(object)" 
class Person(object): 
    
    # Constructor 
    def __init__(self, name): 
        self.name = name 

    # To get name 
    def getName(self): 
        return self.name 

    # To check if this person is employee 
    def isEmployee(self): 
        return False


# Inherited or Sub class (Note Person in bracket) 
class Employee(Person): 

    # Here we return true 
    def isEmployee(self): 
        return True

# Driver code 
emp = Person("Geek1") # An Object of Person 
print(emp.getName(), emp.isEmployee())  # Geek1 False

emp = Employee("Geek2") # An Object of Employee 
print(emp.getName(), emp.isEmployee())  # Geek2 True

--------------------------------------------------------------------------------

# Python example to check if a class is 
# subclass of another 

class Base(object): 
    pass # Empty Class 

class Derived(Base): 
    pass # Empty Class 

# Driver Code 
print(issubclass(Derived, Base)) # True
print(issubclass(Base, Derived)) # False
 
d = Derived() 
b = Base() 

# b is not an instance of Derived 
print(isinstance(b, Derived))  # False

# But d is an instance of Base 
print(isinstance(d, Base))  # True

--------------------------------------------------------------------------------

# Python example to show that base 
# class members can be accessed in 
# derived class using super() 
class Base(object): 

    # Constructor 
    def __init__(self, x): 
        self.x = x   

class Derived(Base): 

    # Constructor 
    def __init__(self, x, y): 
        
        ''' In Python 3.x, "super().__init__(name)" 
            also works'''
        super(Derived, self).__init__(x) 
        self.y = y 

    def printXY(self): 

    # Note that Base.x won't work here 
    # because super() is used in constructor 
    print(self.x, self.y) 


# Driver Code 
d = Derived(10, 20) 
d.printXY() # (10, 20)

--------------------------------------------------------------------------------

## Multiple Inheritance
============================

# Python example to show working of multiple 
# inheritance 
class Base1(object): 
    def __init__(self): 
        self.str1 = "Geek1"
        print("Base1")

class Base2(object): 
    def __init__(self): 
        self.str2 = "Geek2"     
        print("Base2")

class Derived(Base1, Base2): 
    def __init__(self): 
        
        # Calling constructors of Base1 
        # and Base2 classes 
        Base1.__init__(self) 
        Base2.__init__(self) 
        print("Derived")
        
    def printStrs(self): 
        print(self.str1, self.str2) 
        

ob = Derived() 
# Base1
# Base2
# Derived
ob.printStrs() 
# Geek1 Geek2
--------------------------------------------------------------------------------


class X(object): 
    def __init__(self, a): 
        self.num = a 
    def doubleup(self): 
        self.num *= 2

class Y(X): 
    def __init__(self, a): 
        X.__init__(self, a) 
    def tripleup(self): 
        self.num *= 3

obj = Y(4) 
print(obj.num)  # 4

obj.doubleup() 
print(obj.num)  # 8

obj.tripleup() 
print(obj.num)  # 24


--------------------------------------------------------------------------------


# Base or Super class 
class Person(object): 
    def __init__(self, name): 
        self.name = name 
        
    def getName(self): 
        return self.name 
    
    def isEmployee(self): 
        return False

# Inherited or Subclass (Note Person in bracket) 
class Employee(Person): 
    def __init__(self, name, eid): 

        ''' In Python 3.0+, "super().__init__(name)" 
            also works'''
        super(Employee, self).__init__(name) 
        self.empID = eid 
        
    def isEmployee(self): 
        return True
        
    def getID(self): 
        return self.empID 

# Driver code 
emp = Employee("Geek1", "E101") 
print(emp.getName(), emp.isEmployee(), emp.getID()) #('Geek1', True, 'E101')

--------------------------------------------------------------------------------

## Private Variable & Methods
====================================

class MyClass: 

    # Hidden member of MyClass 
    __hiddenVariable = 0
    
    # A member method that changes 
    # __hiddenVariable 
    def add(self, increment): 
        self.__hiddenVariable += increment 
        print (self.__hiddenVariable) 

# Driver code 
myObject = MyClass()     
myObject.add(2) # 2
myObject.add(5) # 7

# AttributeError: MyClass instance has no attribute '__hiddenVariable'
print (myObject.__hiddenVariable) 


--------------------------------------------------------------------------------


# A Python program to demonstrate that hidden 
# members can be accessed outside a class 
class MyClass: 

    # Hidden member of MyClass 
    __hiddenVariable = 10

# Driver code 
myObject = MyClass()     
print(myObject._MyClass__hiddenVariable)  # 10

   
--------------------------------------------------------------------------------

# Private methods are accessible outside their class, just not easily accessible. 
# Nothing in Python is truly private; 
# internally, the names of private methods and attributes are mangled and unmangled on 
# the fly to make them seem inaccessible by their given names

class Test: 
    def __init__(self, a, b): 
        self.a = a 
        self.b = b 

    def __repr__(self): 
        return "Test a:%s b:%s" % (self.a, self.b) 

    def __str__(self): 
        return "From str method of Test: a is %s,""b is %s" % (self.a, self.b) 

# Driver Code        
t = Test(1234, 5678) 
print(t) # This calls __str__() -> From str method of Test: a is 1234,b is 5678
print([t]) # This calls __repr__() -> [Test a:1234 b:5678]


--------------------------------------------------------------------------------

# If no __str__ method is defined, print t (or print str(t)) uses __repr__. 

class Test: 
    def __init__(self, a, b): 
        self.a = a 
        self.b = b 

    def __repr__(self): 
        return "Test a:%s b:%s" % (self.a, self.b) 

# Driver Code        
t = Test(1234, 5678) 
print(t) # Test a:1234 b:5678



# If no __repr__ method is defined then the default is used.

class Test: 
    def __init__(self, a, b): 
        self.a = a 
        self.b = b 

# Driver Code        
t = Test(1234, 5678) 
print(t) # <__main__.Test instance at 0x7fa079da6710>


--------------------------------------------------------------------------------


def Merge(dict1, dict2): 
    res = {**dict1, **dict2} 
    return res 
    
# Driver code 
dict1 = {'a': 10, 'b': 8} 
dict2 = {'d': 6, 'c': 4} 
dict3 = Merge(dict1, dict2) 
print(dict3)                # {'a': 10, 'b': 8, 'd': 6, 'c': 4}


# Python program to illustrate 
# *args with first extra argument 
def myFun(arg1, *argv): 
    print ("First argument :", arg1) 
    for arg in argv: 
        print("Next argument through *argv :", arg) 

myFun('Hello', 'Welcome', 'to', 'GeeksforGeeks') 

# First argument : Hello
# Next argument through *argv : Welcome
# Next argument through *argv : to
# Next argument through *argv : GeeksforGeeks

--------------------------------------------------------------------------------


## Dictionary Methods
=======================
# Python program to illustrate **kargs for 
# variable number of keyword arguments with 
# one extra argument. 

def myFun(arg1, **kwargs): 
    for key, value in kwargs.items(): 
        print ("%s == %s" %(key, value)) 

# Driver code 
myFun("Hi", first ='Geeks', mid ='for', last='Geeks')    

def myFun(arg1, arg2, arg3): 
    print("arg1:", arg1) 
    print("arg2:", arg2) 
    print("arg3:", arg3) 
    
# Now we can use *args or **kwargs to 
# pass arguments to this function : 
args = ("Geeks", "for", "Geeks") 
myFun(*args) 
# first == Geeks
# mid == for
# last == Geeks
# arg1: Geeks
# arg2: for
# arg3: Geeks

kwargs = {"arg1" : "Geeks", "arg2" : "for", "arg3" : "Geeks"} 
myFun(**kwargs) 
# first == Geeks
# mid == for
# last == Geeks
# arg1: Geeks
# arg2: for
# arg3: Geeks

--------------------------------------------------------------------------------

# Python code to merge dict using update() method 
def Merge(dict1, dict2): 
    return(dict2.update(dict1)) 
    
# Driver code 
dict1 = {'a': 10, 'b': 8} 
dict2 = {'d': 6, 'c': 4} 

# This return None 
print(Merge(dict1, dict2)) # None

# changes made in dict2 
print(dict2) # {'d': 6, 'c': 4, 'a': 10, 'b': 8}


# Python code to merge dict using a single 
# expression 


print("=================== package ================================")
'''
file name : first.py
x = 9
def setx(n):
    global x
    x = n
    print(x)
'''
import first
#first.setx(8)

x = 11
def f1():
    x = 22
    print("inside f1 %d"%x)
    def f2():
        print("inside f2 %d"%x)
    return f2
var = f1()
var()
