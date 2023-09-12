============|
Exception   |
============|
How try() works?
----------------------
First try clause is executed i.e. the code between try and except clause.
If there is no exception, then only try clause will run, except clause is finished.
If any exception occured, try clause will be skipped and except clause will run.
If any exception occurs, but the except clause within the code doesn’t handle it, 
it is passed on to the outer try statements. If the exception left unhandled, then the execution stops.
A try statement can have more than one except clause

Code 1 : No exception, so try clause will run.
---------------------------------------------------

# Python code to illustrate 
# working of try() 
def divide(x, y): 
	try: 
		# Floor Division : Gives only Fractional Part as Answer 
		result = x // y 
		print("Yeah ! Your answer is :", result) 
	except ZeroDivisionError: 
		print("Sorry ! You are dividing by zero ") 

# Look at parameters and note the working of Program 
divide(3, 2) 

Output :

		('Yeah ! Your answer is :', 1)

Code 1 : There is an exception so only except clause will run.
-------------------------------------------------------------------
# Python code to illustrate 
# working of try() 
def divide(x, y): 
	try: 
		# Floor Division : Gives only Fractional Part as Answer 
		result = x // y 
		print("Yeah ! Your answer is :", result) 
	except ZeroDivisionError: 
		print("Sorry ! You are dividing by zero ") 

# Look at parameters and note the working of Program 
divide(3, 0) 

Output :

		Sorry ! You are dividing by zero


========|
Error   |
========|
# Python program to handle simple runtime error 

a = [1, 2, 3] 
try: 
	print "Second element = %d" %(a[1]) 

	# Throws error since there are only 3 elements in array 
	print "Fourth element = %d" %(a[3]) 

except IndexError: 
	print "An error occurred"

Output:

		Second element = 2
		An error occurred

# Program to handle multiple errors with one except statement 
try : 
	a = 3
	if a < 4 : 

		# throws ZeroDivisionError for a = 3 
		b = a/(a-3) 
	
	# throws NameError if a >= 4 
	print "Value of b = ", b 

# note that braces () are necessary here for multiple exceptions 
except(ZeroDivisionError, NameError): 
	print "\nError Occurred and Handled"

Output:

		Error Occurred and Handled


# Program to depict else clause with try-except 

# Function which returns a/b 
def AbyB(a , b): 
	try: 
		c = ((a+b) / (a-b)) 
	except ZeroDivisionError: 
		print "a/b result in 0"
	else: 
		print c 

# Driver program to test above function 
AbyB(2.0, 3.0) 
AbyB(3.0, 3.0) 

Output:
		-5.0
		a/b result in 0

# Python program to demonstrate finally 
	
# No exception Exception raised in try block 
try: 
	k = 5//0 # raises divide by zero exception. 
	print(k) 
	
# handles zerodivision exception	 
except ZeroDivisionError:	 
	print("Can not divide by zero") 
		
finally: 
	# this block is always executed 
	# regardless of exception generation. 
	print('This is always executed') 

Output:

		Can not divide by zero
		This is always executed

Raising Exception
---------------------------
The raise statement allows the programmer to force a specific exception to occur. 
The sole argument in raise indicates the exception to be raised. 
This must be either an exception instance or an exception class (a class that derives from Exception).

# Program to depict Raising Exception 

try: 
	raise NameError("Hi there") # Raise Error 
except NameError: 
	print("An exception")
	raise # To determine whether the exception was raised or not 

The output of the above code will simply line printed as “An exception” but a Runtime 
error will also occur in the last due to raise statement in the last line. 
So, the output on your command line will look like

Traceback (most recent call last):
  File "003dff3d748c75816b7f849be98b91b8.py", line 4, in 
    raise NameError("Hi there") # Raise Error
NameError: Hi there

Creating User-defined Exception
----------------------------------------
Programmers may name their own exceptions by creating a new exception class. 
Exceptions need to be derived from the Exception class, either directly or indirectly. Although not mandatory, 
most of the exceptions are named as names that end in “Error” similar to naming of the standard exceptions in python. 

# A python program to create user-defined exception 

# class MyError is derived from super class Exception 
class MyError(Exception): 

	# Constructor or Initializer 
	def __init__(self, value): 
		self.value = value 

	# __str__ is to print() the value 
	def __str__(self): 
		return(repr(self.value)) 

try: 
	raise(MyError(3*2)) 

# Value of Exception is stored in error 
except MyError as error: 
	print('A New Exception occured: ',error.value) 


Output:

		('A New Exception occured: ', 6)

To know more about about class Exception, ------> help(Exception) 

 Deriving Error from Super Class Exception

Super class Exceptions are created when a module needs to handle several distinct errors. 
One of the common way of doing this is to create a base class for exceptions defined by that module. 
Further, various subclasses are defined to create specific exception classes for different error conditions.

# class Error is derived from super class Exception 
class Error(Exception): 

	# Error is derived class for Exception, but 
	# Base class for exceptions in this module 
	pass

class TransitionError(Error): 

	# Raised when an operation attempts a state 
	# transition that's not allowed. 
	def __init__(self, prev, nex, msg): 
		self.prev = prev 
		self.next = nex 

		# Error message thrown is saved in msg 
		self.msg = msg 
try: 
	raise(TransitionError(2,3*2,"Not Allowed")) 

# Value of Exception is stored in error 
except TransitionError as error: 
	print('Exception occured: ',error.msg) 

Output:

		('Exception occured: ', 'Not Allowed')


How to use standard Exceptions as base class?
----------------------------------------------------------
Runtime error is a class is a standard exception which is raised when a generated error does not fall into any category. 
This program illustrates how to use runtime error as base class and network error as derived class. 
In a similar way, any exception can be derived from the standard exceptions of Python.


# NetworkError has base RuntimeError 
# and not Exception 
class Networkerror(RuntimeError): 
	def __init__(self, arg): 
		self.args = arg 

try: 
	raise Networkerror("Error") 

except Networkerror as e: 
	print (e.args) 

Output:

		('E', 'r', 'r', 'o', 'r')


=====================|
Clean Up Actions     |
=====================|
 Code, raise error but we don’t have any except clause to handle it. 
 So, clean-up action is taken first and then the error(by default) is raised by the compiler.

# Python code to illustrate 
# clean up actions 
def divide(x, y): 
	try: 
		# Floor Division : Gives only Fractional Part as Answer 
		result = x // y 
	except ZeroDivisionError: 
		print("Sorry ! You are dividing by zero ") 
	else: 
		print("Yeah ! Your answer is :", result) 
	finally: 
		print("I'm finally clause, always raised !! ") 

# Look at parameters and note the working of Program 
divide(3, "3") 

Output :

		I am finally clause, always raised !! 

		Error:

		Traceback (most recent call last):
		  File "C:/Users/DELL/Desktop/Code.py", line 15, in 
		    divide(3, "3")
		  File "C:/Users/DELL/Desktop/Code.py", line 7, in divide
		    result = x // y
		TypeError: unsupported operand type(s) for //: 'int' and 'str'



