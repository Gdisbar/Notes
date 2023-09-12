class A:
    class_instance = 15

    def __init__(self,object_instance,time):
        self.object_instance = object_instance
        self.time = time

    def show(self):
        print(self.object_instance , self.time,self.class_instance)

    @classmethod    
    def print(cls):
        return cls.class_instance

    @staticmethod
    def info():
        return 'This is Static method'  

a=A("Raju",3)
a.show()                                                        # Raju 3 15
b=A("Raju",3)
b.time = 4
print("============================ modified class_instance ===============================")
b.show()                                                        # Raju 4 15
print('class_instance before updation ',A.class_instance)       # 15
A.class_instance = 23
print('class instance value updated %d'%A.class_instance)       # 23
#A("test class A").display()
print('class method returns {}'.format(A.print()))              # 23
print('static method returns ===>',A.info())                    # This is Static method


## Another Example - Class Method & Static method

class Employee:

    num_of_emps = 0
    raise_amt = 1.04

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.email = first + '.' + last + '@email.com'
        self.pay = pay

        Employee.num_of_emps += 1

    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amt)

    @classmethod
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @classmethod
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

    @staticmethod
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        return True


emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('Test', 'Employee', 60000)

Employee.set_raise_amt(1.05)

print(Employee.raise_amt) #1.05
print(emp_1.raise_amt)    #1.05
print(emp_2.raise_amt)    #1.05

emp_str_1 = 'John-Doe-70000'
emp_str_2 = 'Steve-Smith-30000'
emp_str_3 = 'Jane-Doe-90000'

first, last, pay = emp_str_1.split('-')

#new_emp_1 = Employee(first, last, pay)
new_emp_1 = Employee.from_string(emp_str_1)

print(new_emp_1.email) #John.Doe@email.com
print(new_emp_1.pay)   #70000

import datetime
my_date = datetime.date(2016, 7, 11)

print(Employee.is_workday(my_date)) #True

print("================================= sub class =========================================")

class student:

    def __init__(self,name,roll):
        self.name = name
        self.roll = roll
        self.lap = self.laptop()

    def show(self):
        print(self.name,self.roll)  

    def show(self):
        print(self.name,self.roll,self.laptop().ram,self.laptop().brand)


    class laptop:
    
        def __init__(self):
            self.brand = 'ACER'
            self.ram = 12   

        def show(self):
            print(self.brand,self.ram)

s1=student('gorgio',29)
s1.show()               # gorgio 29
lap1 = s1.laptop()
lap1.ram = 8
lap1.show()             # ACER 8
s1.show()               # gorgio 29 12 ACER


print("============================================ polymorphism ========================")
print("==================== duck typing ================================")
class Dog:
    def talk(self):
        print("dog is barking")

class Cat:
    def talk(self):
        print("cat is mewing")

class Animal:
    def alive(self,species):
        species.talk()

d=Dog()
c=Cat()

live = Animal()         
live.alive(d)     # dog is barking
live.alive(c)     # cat is mewing

print("=============== operator overloading ============================")
a = 10
b = 58
print('using direct call ',int.__mul__(a,b))

class A:
	def __init__(self,x,y):
		self.x = x
		self.y = y

	def __add__(self,other):
		x = self.x + other.x
		y = self.y + other.y
		z=A(x,y)
		return z

	def __mul__(self,other):
		x=self.x * other.x - self.y * other.y
		y=self.y * other.x + self.x * other.y
		zm=A(x,y)
		return zm

	def __gt__(self,other):
		x = self.x + self.y	
		y = other.x + other.y

		if x > y:
			return True
		else:
			return False	

	#def __str__(self):
	#	return ('{} ,{}'.format(self.x,self.y))		

a1=A(2,3)
a2=A(4,5)
a3 = a1 + a2
print('complex __add__ result : {}+{}j'.format(a3.x,a3.y))
a4 = a1 * a2
print('complex __mul__ result : {}+{}j'.format(a4.x,a4.y))		

if a1 > a2:
	print('a1 is grater ({} , {})'.format(a1.x,a2.y))
else:
	print('a2 is grater ({} , {})'.format(a2.x,a2.y))	
#print('without overloading',a1)
m =-553
print(m.__str__())
print('after overloading ',a1.__str__())	#uncomment __str__() to print values

print("======================== method overloading ===================================")

class N:
    def sum(self,a=None,b=None,c=None):
        if a!=None and b!=None and c!=None:
            return a+b+c
        elif a!=None and b!=None:
            return a+b  
        elif a!=None:
            return a
        else:
            return 0

n=N()
print(n.sum())     # 0
print(n.sum(1))    # 1          
print(n.sum(1,2))  # 3
print(n.sum(1,2,3))# 6

print("======================== method overriding ===================================")

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

    def __repr__(self):
        return "Employee('{}', '{}', {})".format(self.first, self.last, self.pay)

    def __str__(self):
        return '{} - {}'.format(self.fullname(), self.email)

    def __add__(self, other):
        return self.pay + other.pay

    def __len__(self):
        return len(self.fullname())


emp_1 = Employee('Corey', 'Schafer', 50000)
emp_2 = Employee('Test', 'Employee', 60000)

# print(emp_1 + emp_2)

print(len(emp_1)) # 13

class B:
    def show(self):
        print("this is class B")
        
class C(B):
    pass
    # def show(self):
    #   print("this is class C")

c = C()
c.show()    ##uncomment show method of C
# this is class B
# this is class C -> after uncommenting show method of C

print("========================= inherite =================================================================")



class A():

    def __init__(self):
        print('class A constructor')

    def fun(self):
        print('class A ')

class B(A):

    def __init__(self):
        super().__init__()
        print('class B constructor')

    def fun2(self):
        print('class B')
            
class C():
    def fun(self):
        print('class C ')

class D(A,C):

    def __init__(self):
        super().__init__()      #method resolution order
        print('class C constructor')

    def fun4(self):
        print('class D ')   

    def fun(self):
        print('class A-D,race condition')       

class E(B):
    
    def __init__(self):
        super().fun2()  

print("B() :: ",B()) 
# class A constructor
# class B constructor
# B() ::  <__main__.B object at 0x7f22daa92cb0>

print("b.fun() :: ",B().fun())
# class A constructor
# class B constructor
# class A 
# b.fun() ::  None

print("b.fun2() :: ",B().fun2())
# class A constructor
# class B constructor
# class B
# b.fun2() ::  None

print("D() :: ",D())
# class A constructor
# class C constructor
# D() ::  <__main__.D object at 0x7fcc8bb96cb0>

print("d.fun() :: ",D().fun()) #biased towards fun of class A
# class A constructor
# class C constructor
# class A-D,race condition
# d.fun() ::  None

print("d.fun4() :: ",D().fun4()) 
# class A constructor
# class C constructor
# class D 
# d.fun4() ::  None

print("E() :: ",E())
# class B
# E() ::  <__main__.E object at 0x7fd988c92cb0>

print("e.fun2() :: ",E().fun2())
# class B
# class B
# e.fun2() ::  None

print("e.fun() :: ",E().fun())
# class B
# class A 
# e.fun() ::  None


print("============================= instance ========================================")
class FirstClass:
    def setdata(self,data):
        self.data = data

    def getdata(self):
        return self.data

class SecondClass(FirstClass):
    def __init__(self,data):
        self.data = data

    def __add__(self,other):
        return (self.data + other)

    def __str__(self):
        return ' %s' %self.data     
    

x = FirstClass()
x.setdata("python")
print('printing x %s'%x.getdata()) # python

y = SecondClass('abc')
print('printing y %s'%y)           # abc
print(y.getdata())                 # abc
z = y + 'xyz'
print('printing z  %s'%z)          # abcxyz

'''
# cook your dish here
for _ in range(int(input())):
    k = 0
    for i in range(int(input())):
        id, s = map(int, input().split())
        k += (id - s)
    print(k)
'''
--------------------------------------------------------------------------------------------------------------------------
## Decorators
class Employee:

    def __init__(self, first, last):
        self.first = first
        self.last = last

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    @fullname.setter
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last
    
    @fullname.deleter
    def fullname(self):
        print('Delete Name!')
        self.first = None
        self.last = None


emp_1 = Employee('John', 'Smith')
emp_1.fullname = "Corey Schafer"

print(emp_1.first) # Corey
print(emp_1.email) # Corey.Schafer@email.com
print(emp_1.fullname) # Corey Schafer

del emp_1.fullname # Delete Name!

print("==================== function ======================================")
print("==================== returning multiple value from function ============================")
def add_mul(x,y):
	return x+y,x*y
m,n = add_mul(2,5)
print(m,n)	

print("==================== variable length args ================================================")
def num(*n):
	x = 0
	for i in n:
		x+=i
	print(x)
	
num(2,3,5,2)	# 12	

print("==================== variable length args(some args are defined) ==========================")
def id(name,**args):
	print(name)
	print(args)

id(name="Jotin",age=12,gender='M',visited=True)	

# Jotin
# {'age': 12, 'gender': 'M', 'visited': True}

print("====================== return type is function ============================================")
def div(a,b):
	print(a/b)

def smart_div(func):

	def inner(a,b):
		if a<b:
			a,b=b,a
		return func(a,b)
	return inner

div1 = smart_div(div)
div1(2,4)	#   # 2.0 		


