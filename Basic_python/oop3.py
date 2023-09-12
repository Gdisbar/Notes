
=====================|
Garbage Collection   |
=====================|
x = [] 
x.append(l) 
x.append(2) 

# delete the list from memory or 
# assigning object x to None(Null) 
del x 
# x = None 

Automatic Garbage Collection
-----------------------------
# loading gc 
import gc 

# get the current collection 
# thresholds as a tuple 
print("Garbage collection thresholds:", gc.get_threshold())

Output:

		Garbage collection thresholds: (700, 10, 10) 


Manual Garbage Collection
---------------------------
# Importing gc module 
import gc 

# Returns the number of 
# objects it has collected 
# and deallocated 
collected = gc.collect() 

# Prints Garbage collector 
# as 0 object 
print("Garbage collector: collected", "%d objects." % collected) 


import gc 
i = 0

# create a cycle and on each iteration x as a dictionary 
# assigned to 1 
def create_cycle(): 
	x = { } 
	x[i+1] = x 
	print(x) 

# lists are cleared whenever a full collection or 
# collection of the highest generation (2) is run 
collected = gc.collect() # or gc.collect(2) 
print("Garbage collector: collected %d objects." % (collected) )

print ("Creating cycles...")
for i in range(10): 
	create_cycle() 

collected = gc.collect() 

print ("Garbage collector: collected %d objects." % (collected) )

Output:

		Garbage collector: collected 0 objects.
		Creating cycles...
		{1: {...}}
		{2: {...}}
		{3: {...}}
		{4: {...}}
		{5: {...}}
		{6: {...}}
		{7: {...}}
		{8: {...}}
		{9: {...}}
		{10: {...}}
		Garbage collector: collected 10 objects.


===============|
Reflection     |
===============|

Reflection refers to the ability for code to be able to examine attributes about objects 
that might be passed as parameters to a function. 
For example, if we write type(obj) then Python will return an object which represents the 
type of obj.

Using reflection, we can write one recursive reverse function that will work for strings, 
lists, and any other sequence that supports slicing and concatenation. 
If an obj is a reference to a string, then Python will return the str type object. 
Further, if we write str() we get a string which is the empty string. In other words, 
writing str() is the same thing as writing “”. 
Likewise, writing list() is the same thing as writing [].

# Python program to illustrate reflection 
def reverse(seq): 
	SeqType = type(seq) 
	emptySeq = SeqType() 
	
	if seq == emptySeq: 
		return emptySeq 
	
	restrev = reverse(seq[1:]) 
	first = seq[0:1] 
	
	# Combine the result 
	result = restrev + first 
	
	return result 

# Driver code 
print(reverse([1, 2, 3, 4])) 
print(reverse("HELLO")) 

Output:

		[4,3,2,1]
		OLLEH


Reflection-enabling functions
------------------------------------------
Reflection-enabling functions include type(), isinstance(), callable(), dir() and getattr().

Callable() :A callable means anything that can be called. For an object, determines whether 
it can be called. A class can be made callable by providing a __call__() method. 
The callable() method returns True if the object passed appears callable. 
If not, it returns False.
   

x = 5

def testFunction():
  print("Test")
   
y = testFunction

if (callable(x)):
    print("x is callable")
else:
    print("x is not callable")

if (callable(y)):
    print("y is callable")
else:
    print("y is not callable")

Output:

	    x is not callable
	    y is callable

 

class Foo1:
  def __call__(self):
    print('Print Something')

print(callable(Foo1))

Output:

		True

Dir : The dir() method tries to return a list of valid attributes of the object. 
The dir() tries to return a list of valid attributes of the object.
If the object has __dir__() method, the method will be called and must return the list 
of attributes. If the object doesn’t have __dir()__ method, this method tries to find 
information from the __dict__ attribute (if defined), and from type object. 
In this case, the list returned from dir() may not be complete. 


number = [1,2,3]
print(dir(number))

characters = ["a", "b"]
print(dir(characters))



Getattr : The getattr() method returns the value of the named attribute of an object. 
If not found, it returns the default value provided to the function.The getattr method 
takes three parameters object, name and default(optional).

class Employee:
    salary = 25000
    company_name= "geeksforgeeks"

employee = Employee()
print('The salary is:', getattr(employee, "salary"))
print('The salary is:', employee.salary)

Output:

		The salary is: 25000
		The salary is: 25000



================================|
Class & Instance Attributes     |
================================|
Class attributes belong to the class itself they will be shared by all the instances. 
Such attributes are defined in the class body parts usually at the top, for legibility.

# Write Python code here 
class sampleclass: 
	count = 0	 # class attribute 

	def increase(self): 
		sampleclass.count += 1

# Calling increase() on an object 
s1 = sampleclass() 
s1.increase()		 
print(s1.count) 

# Calling increase on one more 
# object 
s2 = sampleclass() 
s2.increase() 
print(s2.count) 

print(sampleclass.count) 

Output:

		1              
		2                           
		2

Unlike class attributes, instance attributes are not shared by objects. Every object 
has its own copy of the instance attribute (In case of class attributes all object refer 
to single copy).


To list the attributes of an instance/object, we have two functions:-
1. vars()– This function displays the attribute of an instance in the form of an dictionary.
2. dir()– This function displays more attributes than vars function,as it is not limited to 
	instance. 
It displays the class attributes as well. It also displays the attributes of its ancestor classes.

# Python program to demonstrate 
# instance attributes. 
class emp: 
	def __init__(self): 
		self.name = 'xyz'
		self.salary = 4000

	def show(self): 
		print(self.name) 
		print(self.salary) 

e1 = emp() 
print("Dictionary form :", vars(e1)) 
print(dir(e1)) 

Output :

		Dictionary form :{'salary': 4000, 'name': 'xyz'}
		['__doc__', '__init__', '__module__', 'name', 'salary', 'show']


==============================|
Meta-programming/Meta-Class   |
==============================|
Every type in Python is defined by Class. So in above example, unlike C or Java where int, char, 
float are primary data types, in Python they are object of int class or str class. 
So we can make a new type by creating a class of that type. 
For example we can create a new type Student by creating Student class.

class Student: 
	pass
stu_obj = Student() 

# Print type of object of Student class 
print("Type of stu_obj is:", type(stu_obj)) 

Output:

		Type of stu_obj is: <class '__main__.Student'>


A Class is also an object, and just like any other object it’s a instance of something called Metaclass. 
A special class type creates these Class object. The type class is default metaclass which is responsible for making classes. 
For example in above example if we try to find out the type of Student class, it comes out to be a type.

class Student: 
	pass

# Print type of Student class 
print("Type of Student class is:", type(Student)) 

Output:

		Type of Student class is: <class 'type'>

Because Classes are also an object, they can be modified in same way. 
We can add or subtract fields or methods in class in same way we did with other objects.

# Defined class without any 
# class methods and variables 
class test:pass

# Defining method variables 
test.x = 45

# Defining class methods 
test.foo = lambda self: print('Hello') 

# creating object 
myobj = test() 

print(myobj.x) 
myobj.foo() 

Output:

		45
		Hello

This whole meta thing can be summarized as – Metaclass create Classes and Classes creates objects
-------------------------------------------------------------------------------------------------------

Object ----(instance of)-----------> class -------(instance of)-----> metaclass

Metaclass is responsible for generation of classes, so we can write our own custom metaclasses to modify 
the way classes are generated by performing extra actions or injecting code. Usually we do not need 
custom metaclasses but sometime it’s necessary.
There are problems for which metaclass and non-metaclass based solutions are available (often simpler) 
but in some cases only metaclass can solve the problem. We will discuss such problem in this article.


Creating custom Metaclass
--------------------------------------------
To create our custom metaclass, our custom metaclass have to inherit type metaclass and usually override –

__new__(): It’s a method which is called before __init__(). It creates the object and return it. 
			We can overide this method to control how the objects are created.
__init__(): This method just initialize the created object passed as parameter

We can create classes using type() function directly. It can be called in following ways –

When called with only one argument, it returns the type. We have seen it before in above examples.
When called with three parameters, it creates a class. Following arguments are passed to it –
    Class name
    Tuple having base classes inherited by class
    Class Dictionary: It serves as local namespace for the class, populated with class methods and variables

def test_method(self): 
	print("This is Test class method!") 

# creating a base class 
class Base: 
	def myfun(self): 
		print("This is inherited method!") 

# Creating Test class dynamically using 
# type() method directly 
Test = type('Test', (Base, ), dict(x="atul", my_method=test_method)) 

# Print type of Test 
print("Type of Test class: ", type(Test)) 

# Creating instance of Test class 
test_obj = Test() 
print("Type of test_obj: ", type(test_obj)) 

# calling inherited method 
test_obj.myfun() 

# calling Test class method 
test_obj.my_method() 

# printing variable 
print(test_obj.x) 

Output:

		Type of Test class:  <class 'type'>
		Type of test_obj:  <class '__main__.Test'>
		This is inherited method!
		This is Test class method!
		atul

Now let’s create a metaclass without using type() directly. In the following example we will be creating a metaclass 
MultiBases which will check if class being created have inherited from more than one base classes. 
If so, it will raise an error.

# our metaclass 
class MultiBases(type): 
	# overriding __new__ method 
	def __new__(cls, clsname, bases, clsdict): 
		# if no of base classes is greator than 1 
		# raise error 
		if len(bases)>1: 
			raise TypeError("Inherited multiple base classes!!!") 
		
		# else execute __new__ method of super class, ie. 
		# call __init__ of type class 
		return super().__new__(cls, clsname, bases, clsdict) 

# metaclass can be specified by 'metaclass' keyword argument 
# now MultiBase class is used for creating classes 
# this will be propagated to all subclasses of Base 
class Base(metaclass=MultiBases): 
	pass

# no error is raised 
class A(Base): 
	pass

# no error is raised 
class B(Base): 
	pass

# This will raise an error! 
class C(A, B): 
	pass


Output:

		Traceback (most recent call last):
		  File "<stdin>", line 2, in <module>
		  File "<stdin>", line 8, in __new__
		TypeError: Inherited multiple base classes!!!

Solving problem with metaclass

There are some problems which can be solved by decorators (easily) as well as by metaclasses. 
But there are few problems whose result can only be achived by metaclasses. 
For example consider a very simple problem of code repetition.

We want to debug class methods, what we want is that whenever class method executes, 
it should print it’s fully qualified name before executing it’s body.


Very first solution that comes in our mind is using method decorators,

from functools import wraps 

def debug(func): 
	'''decorator for debugging passed function'''
	
	@wraps(func) 
	def wrapper(*args, **kwargs): 
		print("Full name of this method:", func.__qualname__) 
		return func(*args, **kwargs) 
	return wrapper 

def debugmethods(cls): 
	'''class decorator make use of debug decorator 
	to debug class methods '''
	
	# check in class dictionary for any callable(method) 
	# if exist, replace it with debugged version 
	for key, val in vars(cls).items(): 
		if callable(val): 
			setattr(cls, key, debug(val)) 
	return cls

# sample class 
@debugmethods
class Calc: 
	def add(self, x, y): 
		return x+y 
	def mul(self, x, y): 
		return x*y 
	def div(self, x, y): 
		return x/y 
	
mycal = Calc() 
print(mycal.add(2, 3)) 
print(mycal.mul(5, 2)) 

Output:

Full name of this method: Calc.add
5
Full name of this method: Calc.mul
10

This solution works fine but there is one problem, what if we want to apply this method decorator to 
all subclasses which inherit this Calc class. In that case we have to separately apply method decorator to 
every subclass just like we did with Calc class.

The problem is if we have many such subclasses, then in that case we won’t like adding 
decorator to each one separately. If we know beforehand that every subclass must have this debug property, 
then we should look up to metaclass based solution.

Have a look at this metaclass based solution, the idea is that classes will be created normally 
and then immediately wrapped up by debug method decorator –

from functools import wraps 

def debug(func): 
	'''decorator for debugging passed function'''
	
	@wraps(func) 
	def wrapper(*args, **kwargs): 
		print("Full name of this method:", func.__qualname__) 
		return func(*args, **kwargs) 
	return wrapper 

def debugmethods(cls): 
	'''class decorator make use of debug decorator 
	to debug class methods '''
	
	for key, val in vars(cls).items(): 
		if callable(val): 
			setattr(cls, key, debug(val)) 
	return cls

class debugMeta(type): 
	'''meta class which feed created class object 
	to debugmethod to get debug functionality 
	enabled objects'''
	
	def __new__(cls, clsname, bases, clsdict): 
		obj = super().__new__(cls, clsname, bases, clsdict) 
		obj = debugmethods(obj) 
		return obj 
	
# base class with metaclass 'debugMeta' 
# now all the subclass of this 
# will have debugging applied 
class Base(metaclass=debugMeta):pass

# inheriting Base 
class Calc(Base): 
	def add(self, x, y): 
		return x+y 
	
# inheriting Calc 
class Calc_adv(Calc): 
	def mul(self, x, y): 
		return x*y 

# Now Calc_adv object showing 
# debugging behaviour 
mycal = Calc_adv() 
print(mycal.mul(2, 3)) 

Output:

		Full name of this method: Calc_adv.mul
		6

When to use Metaclasses
-----------------------------
Most of the time we are not using metaclasses, they are like black magic and usually for something complicated, 
but few cases where we use metaclasses are –

    As we have seen in above example, metaclasses propogate down the inheritance hierarchies. 
    It will affect all the subclasses as well. If we have such situation, then we should use metaclasses.
    If we want to change class automatically, when it is created
    If you are API developer, you might use metaclasses

As quoted by Tim Peters

    Metaclasses are deeper magic that 99% of users should never worry about. If you wonder whether you need them, 
    you don’t (the people who actually need them know with certainty that they need them, 
    and don’t need an explanation about why). 