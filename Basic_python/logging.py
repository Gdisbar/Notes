# Inside test4.py
==================================================================
import logging

# logging.basicConfig(filename='employee.log', level=logging.INFO,
#                     format='%(levelname)s:%(message)s')


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('employee.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class Employee:
    """A sample Employee class"""

    def __init__(self, first, last):
        self.first = first
        self.last = last

        logger.info('Created Employee: {} - {}'.format(self.fullname, self.email))

    @property
    def email(self):
        return '{}.{}@email.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)


emp_1 = Employee('John', 'Smith')
emp_2 = Employee('Corey', 'Schafer')
emp_3 = Employee('Jane', 'Doe')

# Inside employee.log
-------------------------
# INFO:__main__:Created Employee: John Smith - John.Smith@email.com
# INFO:__main__:Created Employee: Corey Schafer - Corey.Schafer@email.com
# INFO:__main__:Created Employee: Jane Doe - Jane.Doe@email.com


## we import Employee class from test4.py - current file is test3.py
===================================================================
import logging
from test4 import Employee


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('sample.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def add(x, y):
    """Add Function"""
    return x + y


def subtract(x, y):
    """Subtract Function"""
    return x - y


def multiply(x, y):
    """Multiply Function"""
    return x * y


def divide(x, y):
    """Divide Function"""
    try:
        result = x / y
    except ZeroDivisionError:
        logger.exception('Tried to divide by zero')
    else:
        return result


num_1 = 10
num_2 = 0

add_result = add(num_1, num_2)
logger.debug('Add: {} + {} = {}'.format(num_1, num_2, add_result))

sub_result = subtract(num_1, num_2)
logger.debug('Sub: {} - {} = {}'.format(num_1, num_2, sub_result))

mul_result = multiply(num_1, num_2)
logger.debug('Mul: {} * {} = {}'.format(num_1, num_2, mul_result))

div_result = divide(num_1, num_2)
logger.debug('Div: {} / {} = {}'.format(num_1, num_2, div_result))

output
-------------------------------------------------------------------
# 2023-01-29 12:43:40,538:__main__:Add: 10 + 0 = 10
# 2023-01-29 12:43:40,538:__main__:Sub: 10 - 0 = 10
# 2023-01-29 12:43:40,538:__main__:Mul: 10 * 0 = 0
# 2023-01-29 12:43:40,538:__main__:Tried to divide by zero
# Traceback (most recent call last):
#   File "/home/acr00/Program/TestFiles/test3.py", line 71, in divide
#     result = x / y
# ZeroDivisionError: division by zero
# 2023-01-29 12:43:40,540:__main__:Div: 10 / 0 = None

# Inside sample.log
------------------------------------------------------------
# 2023-01-29 12:43:40,538:__main__:Tried to divide by zero
# Traceback (most recent call last):
#   File "/home/acr00/Program/TestFiles/test3.py", line 71, in divide
#     result = x / y
# ZeroDivisionError: division by zero
