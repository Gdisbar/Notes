# Functools module is for higher-order functions that work on other functions. 
# It provides functions for working with other functions and callable objects to use or 
# extend them without completely rewriting them. This module has two classes – partial and partialmethod.


from functools import partial
 
 
def power(a, b):
    return a**b
 
 
# partial functions
pow2 = partial(power, b=2)
pow4 = partial(power, b=4)
power_of_5 = partial(power, 5)
 
print(power(2, 3))    # 8
print(pow2(4))        # 16
print(pow4(3))        # 81
print(power_of_5(2))  # 25
 
print('Function used in partial function pow2 :', pow2.func) #<function power at 0x7fb274415510>
print('Default keywords for pow2 :', pow2.keywords)  # {'b': 2}
print('Default arguments for power_of_5 :', power_of_5.args) # (5,)

-----------------------------------------------------------------------------------------------------------------------------

from functools import partialmethod

class Demo:
    def __init__(self):
        self.color = 'black'
 
    def _color(self, type):
        self.color = type
 
    set_red = partialmethod(_color, type='red')
    set_blue = partialmethod(_color, type='blue')
    set_green = partialmethod(_color, type='green')
 
 
obj = Demo()
print(obj.color) # black
obj.set_blue()
print(obj.color) # blue

-----------------------------------------------------------------------------------------------------------------------------

from functools import cmp_to_key

# function to sort according to last character
def cmp_fun(a, b):
    if a[-1] > b[-1]:
        return 1
    elif a[-1] < b[-1]:
        return -1
    else:
        return 0
 
list1 = ['geeks', 'for', 'geeks']
l = sorted(list1, key = cmp_to_key(cmp_fun))
print('sorted list :', l) # ['for', 'geeks', 'geeks']

-----------------------------------------------------------------------------------------------------------------------------
from functools import reduce
list1 = [2, 4, 7, 9, 1, 3]
sum_of_list1 = reduce(lambda a, b:a + b, list1)
 
list2 = ["abc", "xyz", "def"]
max_of_list2 = reduce(lambda a, b:a if a>b else b, list2)
 
print('Sum of list1 :', sum_of_list1) # 26
print('Maximum of list2 :', max_of_list2) # xyz

-----------------------------------------------------------------------------------------------------------------------------
from functools import total_ordering
 
@total_ordering
class N:
    def __init__(self, value):
        self.value = value
    def __eq__(self, other):
        return self.value == other.value
 
    # Reverse the function of
    # '<' operator and accordingly
    # other rich comparison operators
    # due to total_ordering decorator
    def __lt__(self, other):
        return self.value > other.value
 
 
print('6 > 2 :', N(6)>N(2)) # False
print('3 < 1 :', N(3)<N(1))  # True
print('2 <= 7 :', N(2)<= N(7)) # False
print('9 >= 10 :', N(9)>= N(10)) # True
print('5 == 5 :', N(5)== N(5))   # True

-----------------------------------------------------------------------------------------------------------------------------
from functools import update_wrapper, partial
 
def power(a, b):
    ''' a to the power b'''
    return a**b
 
# partial function
pow2 = partial(power, b = 2)
pow2.__doc__='''a to the power 2'''
pow2.__name__ = 'pow2'
 
print('Before wrapper update -')
print('Documentation of pow2 :', pow2.__doc__) # a to the power 2
print('Name of pow2 :', pow2.__name__) # pow2
print()
update_wrapper(pow2, power)
print('After wrapper update -')
print('Documentation of pow2 :', pow2.__doc__) # a to the power b
print('Name of pow2 :', pow2.__name__)  # power


-----------------------------------------------------------------------------------------------------------------------------
from functools import wraps
 
def decorator(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        """Decorator's docstring"""
        return f(*args, **kwargs)
 
    print('Documentation of decorated :', decorated.__doc__) # f's Docstring
    return decorated
 
@decorator
def f(x):
    """f's Docstring"""
    return x
 
print('f name :', f.__name__)    # f
print('Documentation of f :', f.__doc__) # f's Docstring

-----------------------------------------------------------------------------------------------------------------------------

from functools import lru_cache
 
@lru_cache(maxsize = None)
def factorial(n):
    if n<= 1:
        return 1
    return n * factorial(n-1)
print([factorial(n) for n in range(7)]) # [1, 1, 2, 6, 24, 120, 720]
print(factorial.cache_info()) # CacheInfo(hits=5, misses=7, maxsize=None, currsize=7)

-----------------------------------------------------------------------------------------------------------------------------

from functools import singledispatch
 
 
@singledispatch
def fun(s):
    print(s)
 
 
@fun.register(int)
def _(s):
    print(s * 2)
 
 
fun('GeeksforGeeks') # GeeksforGeeks
fun(10) # 20
