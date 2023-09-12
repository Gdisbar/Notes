Infinite iterators # can be replaced with lambda function
----------------------------
------------------------------
Combinatoric iterators 
----------------------------
------------------------------
# product(iterator,repeat) -> cartesian product
-----------------------------------------------------
print(list(product([1, 2], repeat = 2)))   
# [(1, 1), (1, 2), (2, 1), (2, 2)]
print(list(product(['geeks', 'for', 'geeks'], '2'))) 
# [('geeks', '2'), ('for', '2'), ('geeks', '2')]   
print(list(product('AB', [3, 4]))) 
# [('A', 3), ('A', 4), ('B', 3), ('B', 4)]

# permutations(iterator,repeat)  
----------------------------------------
print (list(permutations([1, 'geeks'], 2)))   
# [(1, 'geeks'), ('geeks', 1)]  
print (list(permutations('AB')))  
# [('A', 'B'), ('B', 'A')]    
print(list(permutations(range(3), 2)))  
# [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

# combinations(iterator,repeat) -> sorted order(without replacement)
-------------------------------------------------------------------------
print(list(combinations(['A', 2], 2)))  
# [('A', 2)] 
print(list(combinations('AB', 2)))  
# [('A', 'B')]  
print(list(combinations(range(2), 1)))  
# [(0, ), (1, )]


# combinations_with_replacement(iterator,repeat) -> sorted order(with replacement)
-------------------------------------------------------------------------------------        
print(list(combinations_with_replacement("AB", 2)))  
# [('A', 'A'), ('A', 'B'), ('B', 'B')] 
print(list(combinations_with_replacement([1, 2], 2)))  
# [(1, 1), (1, 2), (2, 2)] 
print(list(combinations_with_replacement(range(2), 1))) 
# [(0, ), (1, )]
    
Terminating iterators
----------------------------
------------------------------
# accumulate(iter, func) -> default prefix sum
--------------------------------------------------
print (list(itertools.accumulate(li1)))   
print (list(itertools.accumulate(li1, operator.mul)))  
        
# chain(iter1, iter2..) -> print all values in all the iterators
# chain.from_iterable(iter_k) -> here iter_k = [iter1, iter2..]
-----------------------------------------------------------------
print (list(itertools.chain(li1, li2, li3)))
print (list(itertools.chain.from_iterable([li1, li2, li3] ))) 

#compress(iter, selector) -> selector is the bool mask
--------------------------------------------------------------
print (list(itertools.compress('GEEKSFORGEEKS', [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]))) 
# ['G', 'F', 'G']
    

# islice(iterable, start, stop, step)-> selectively prints the values 
----------------------------------------------------------------------- 
# idx    1     3     5    6
li = [2, 4, 5, 7, 8, 10, 20]  
print (list(itertools.islice(li, 1, 6, 2)))  
# [4, 7, 10]

# tee(iterator, count)-> splits the container into a number of iterators 
---------------------------------------------------------------------------
li = [2, 4, 6, 7, 8, 10, 20]  
it = itertools.tee(iter(li), 3)  
for i in range (0, 3):  
    print (list(it[i]))  

# [2, 4, 6, 7, 8, 10, 20]
# [2, 4, 6, 7, 8, 10, 20]
# [2, 4, 6, 7, 8, 10, 20]


li = [[10, 13, 454, 66, 44], [10, 8, 7, 23],[12,23,11],[11],[-1,-1]]
#Find max in each list
op = [functools.reduce(max, sublist) for sublist in li]
# [454, 23, 23, 11, -1]


# zip_longest( iterable1, iterable2, fillval) -> zip 2 lists
--------------------------------------------------------------
print (*(itertools.zip_longest('GesoGes', 'ekfrek', fillvalue ='_' )))  
# ('G', 'e') ('e', 'k') ('s', 'f') ('o', 'r') ('G', 'e') ('e', 'k') ('s', '_')


# Check object is iterable or not
------------------------------------
----------------------------------------
def iterable(obj): 
    try: 
        iter(obj) 
        return True
    except TypeError: 
        return False
  
      
for element in [34, [4, 5], (4, 5), {"a":4}, "dfsdf", 4.5]: 
    print(element, " is iterable : ", iterable(element)) 

# 34  is iterable :  False
# [4, 5]  is iterable :  True
# (4, 5)  is iterable :  True
# {'a': 4}  is iterable :  True
# dfsdf  is iterable :  True
# 4.5  is iterable :  False
