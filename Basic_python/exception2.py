======================|
Built-In Exception    |
======================|
# This returns a dictionary of built-in exceptions, functions and attributes.
locals()['__builtins__']


# Types of Exception
==================================
BaseException
--------------------------------
# This is the base class for all built-in exceptions. It is not meant to be directly 
# inherited by user-defined classes. For, user-defined classes, Exception is used. 
# This class is responsible for creating a string representation of the exception using str() 
# using the arguments passed. An empty string is returned if there are no arguments.

# args : The args are the tuple of arguments given to the exception constructor.
# with_traceback(tb) : This method is usually used in exception handling. 
# This method sets tb as the new traceback for the exception and returns the exception object.


try:
    ...
except SomeException:
    tb = sys.exc_info()[2]
    raise OtherException(...).with_traceback(tb)

Exception
--------------------
# This is the base class for all built-in non-system-exiting exceptions. 
# All user-defined exceptions should also be derived from this class.


ArithmeticError
----------------------------------------------------------------------
------------------------------------------------------------------
FloatingPointError -> when a floating point operation fails
# math.exp(1000)

OverflowError -> arithmetic operation is out of range. Integers raise MemoryError instead of OverflowError. 
OverflowError is sometimes raised for integers that are outside a required range. 
Floating point operations are not checked because of the lack of standardization of floating point exception handling in C.

ZeroDivisionError


try: 
	a = 10/0
	print(a) 
except ArithmeticError: 
		print "This statement is raising an arithmetic exception."
else: 
	print("Success.")

# Output :

# 		This statement is raising an arithmetic exception.



BufferError -> BaseClass for buffer related operations cannot be performed.
LookupError -> BaseClass for key,index is invalid

Concrete exceptions
--------------------------------------
-----------------------------------------

AssertionError ->  assert statement fails

# assert (False, 'The assertion failed')

AttributeError -> non-existent attribute is referenced.

# class Attributes(object): 
# 	pass

# object = Attributes() 
# print(object.attribute)

EOFError-> readline(),input() return empty string for EOF


Other Errors :
-----------------------------------------
GeneratorExit
ImportError
ModuleNotFoundError
IndexError
KeyError -> key not present in dic,list
KeyboardInterrupt
NameError -> local or global name is not found. For example, an unqualified variable name.
RecursionError -> max recursion depth exceed , RuntimeError
NotImplementedError
OSError([arg]) -> system-error like “file not found” or “disk full”
ReferenceError -> weak reference proxy is used to access an attribute of the referent 
after the garbage collection.


import gc 
import weakref 
  
class Foo(object): 
  
    def __init__(self, name): 
        self.name = name 
      
    def __del__(self): 
        print '(Deleting %s)' % self
  
obj = Foo('obj') 
p = weakref.proxy(obj) 
  
print 'BEFORE:', p.name 
obj = None
print 'AFTER:', p.name 

Output :


		BEFORE: obj
		(Deleting )
		AFTER:

		Traceback (most recent call last):
		  File "49d0c29d8fe607b862c02f4e1cb6c756.py", line 17, in 
		    print 'AFTER:', p.name
		ReferenceError: weakly-referenced object no longer exists

RuntimeError
----------------------------
--------------------------------
The RuntimeError is raised when no other exception applies. It returns a string indicating what precisely went wrong.
exception StopIteration
StopIteration ->  raised for next() , __next__() method to signal that all items are produced by the iterator.

SyntaxError -> may occur in import statement,exec() or eval() 

try: 
    print(eval('geeks for geeks'))
except SyntaxError as err: 
    print( 'Syntax error %s (%s-%s): %s' % (err.filename, err.lineno, err.offset, err.text)) 
    print(err)

# Syntax error <string> (1-7): geeks for geeks
# invalid syntax (<string>, line 1)


SystemError -> internal error
SystemExit -> sys.exit() is called & clean-up is done
TypeError -> type mismatch

print(('tuple', ) + 'string') 
# TypeError: can only concatenate tuple (not "str") to tuple 

UnboundLocalError -> UnboundLocalError is a subclass of NameError which is raised when a reference is made to a local 
variable in a function or method, but no value has been assigned to that variable.


def global_name_error(): 
    print(unknown_global_name)
  
def unbound_local(): 
    local_val = local_val + 1
    print(local_val)
  
try: 
    global_name_error() 
except NameError as err: 
    print('Global name error:', err )
  
try: 
    unbound_local() 
except UnboundLocalError as err: 
    print('Local name error:', err)

# Output :

# 		Global name error: global name 'unknown_global_name' is not defined
# 		Local name error: local variable 'local_val' referenced before assignment

UnicodeError -> subclass of ValueError , raised when encode-decode error (write type but wrong value) occurs
print(int('a'))
#ValueError: invalid literal for int() with base 10: 'a'
