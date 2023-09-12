def calculate_time(func): 
	
	# added arguments inside the inner1, 
	# if function takes any arguments, 
	# can be added like this. 
	def inner1(*args, **kwargs): 

		# storing time before function execution 
		begin = time.time() 
		
		func(*args, **kwargs) 

		# storing time after function execution 
		end = time.time() 
		print("Total time taken in : ", func.__name__, end - begin) 

	return inner1 



# this can be added to any function present, in this case to calculate a factorial 
@calculate_time
def factorial(num): 

	# sleep 2 seconds because it takes very less time 
	# so that you can see the actual difference 
	time.sleep(2) 
	print(math.factorial(num)) 

# calling the function. 
factorial(5) 

# 120
# Total time taken in :  factorial 2.0025813579559326
	
## Using class
-----------------------------------------
class CalculateTime:

    def __init__(self,factorial):
        self.factorial = factorial

    def __call__(self,*args,**kwargs):
        start = time.time()
        result = self.factorial(*args, **kwargs)
        print("Time taken for ",self.factorial.__name__,time.time()-start)
        return result


@CalculateTime
def factorial(num): 
    time.sleep(2) 
    return math.factorial(num)

 
print("factorial : ",factorial(5))
# Time taken for  factorial 2.0024938583374023
# factorial :  120
--------------------------------------------------------------------------------------------

def hello_decorator(func): 
	def inner1(*args, **kwargs): 
		
		print("before Execution") 
		
		# getting the returned value 
		returned_value = func(*args, **kwargs) 
		print("after Execution") 
		
		# returning the value to the original frame 
		return returned_value 
		
	return inner1 


# adding decorator to the function 
@hello_decorator
def sum_two_numbers(a, b): 
	print("Inside the function") 
	return a + b 

a, b = 1, 2

# getting the value through return of the function 
print("Sum =", sum_two_numbers(a, b)) 

# before Execution
# Inside the function
# after Execution
# Sum = 3
----------------------------------------------------------------------------
from functools import wraps

def makebold(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return "<b>" + fn(*args, **kwargs) + "</b>"
    return wrapper

def makeitalic(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return "<i>" + fn(*args, **kwargs) + "</i>"
    return wrapper

@makebold
@makeitalic
def hello():
    return "hello world"

@makebold
@makeitalic
def log(s):
    return s

print(hello())        # returns "<b><i>hello world</i></b>"
print(hello.__name__) # with functools.wraps() this returns "hello"
print(log('hello'))   # returns "<b><i>hello</i></b>"
