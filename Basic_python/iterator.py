


========================
Iterator
==============

## Difference between __iter__() and iter()

class X:
    def __iter__(self):
        return 2
x = X()
print(x.__iter__())
# 2
print(iter(x))
# TypeError: iter() returned non-iterator of type 'int'


a custom obj.__iter__() wouldn''t catch that the error and return 2, but iter(obj) throws a type error. 
For custom __iter__() we need to create custom __next__()



iterable_value = 'Geeks'
iterable_obj = iter(iterable_value) 

while True: 
	try: 
		item = next(iterable_obj) 
		print(item) 
	except StopIteration: 
		break

# Output :

# 		G                                                                                                                                                                            
# 		e                                                                                                                                                                            
# 		e                                                                                                                                                                            
# 		k                                                                                                                                                                            
# 		s


class Test: 
	def __init__(self, limit): 
		self.limit = limit 

	def __iter__(self): 
		self.x = 10
		return self

	def __next__(self): 
		x = self.x 
		if x > self.limit: 
			raise StopIteration 
		self.x = x + 1; 
		return x 

for i in Test(15): 
	print(i) 

for i in Test(5): 
	print(i) 


# Output :

# 		10
# 		11
# 		12
# 		13
# 		14
# 		15



=============|
Enumerate    |
=============|
for index,value in enumerate(list,start=5):
	print(index,value)

================|
zip function    |
================|

# Two separate lists 
cars = ["Aston", "Audi", "McLaren"] 
accessories = ["GPS", "Car Repair Kit", 
			"Dolby sound kit"] 

# Combining lists and printing 
for c, a in zip(cars, accessories): 
	print("Car: %s, Accessory required: %s"%(c, a) )

# Output:

# 		Car: Aston, Accessory required: GPS
# 		Car: Audi, Accessory required: Car Repair Kit
# 		Car: McLaren, Accessory required: Dolby sound kit


# Unzip lists 
l1,l2 = zip(*[('Aston', 'GPS'),  
              ('Audi', 'Car Repair'),  
              ('McLaren', 'Dolby sound kit')  
           ]) 
  
# Printing unzipped lists       
print(l1) 
print(l2) 

Output:

		('Aston', 'Audi', 'McLaren')
		('GPS', 'Car Repair', 'Dolby sound kit')

===============|
Generator      |
===============|

def fib(limit): 
    a, b = 0, 1
    while a < limit: 
        yield a 
        a, b = b, a + b 
  

x = fib(5) # generator object
print(x.__next__()) # In Python 2, next() 
print(x.__next__()) 
print(x.__next__()) 
print(x.__next__()) 
print(x.__next__()) 
  
#using loop - in-built iterator 
print("\nUsing for in loop") 
for i in fib(5):  
    print(i) 

# 0
# 1
# 1
# 2
# 3


Advantage of Generator over Iterator : for methods/functions we don’t need to write __next__ and __iter__ methods here
A more practical type of stream processing is handling large data files such as log files. 
Generators provide a space efficient method for such data processing as only parts of the file are 
handled at one given point in time. 
