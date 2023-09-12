T = int(input())
for _ in range(T):
	n,m = map(int,input().strip().split())
	arr = list(map(int,input().strip().split()))
	k = int(input())


import atexit
import io
import sys
from collections import defaultdict
import queue
#Contributed by : Nagendra Jha


#Graph Class:
class Graph():
    def __init__(self,vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self,u,v): # add directed edge from u to v.
        self.graph[u].append(v)

if __name__ == '__main__':
    test_cases = int(input())
    for cases in range(test_cases) :
        N,E = map(int,input().strip().split()) 	#N=vertice,E=edges
        g = Graph(N)
        edges = list(map(int,input().strip().split()))

        for i in range(0,len(edges),2):
            u,v = edges[i],edges[i+1]
            g.addEdge(u,v) # add a directed edge from u to v

        res = bfs(g.graph,N) # print bfs of graph
        for i in range (len (res)):
            print (res[i], end = " ")
        print()




# import sys
# sys.setrecursionlimit(10**6)
# def creategraph(e, n, arr, graph):
#     i = 0
#     while i<2*e:
#         graph[arr[i]].append(arr[i+1])
#         i+=2
        
# def check(graph, N, res):
# 	map=[0]*N
# 	for i in range(N):
# 		map[res[i]]=i
# 	for i in range(N):
# 		for v in graph[i]:
# 			if map[i] > map[v]:
# 				return False
# 	return True

# from collections import defaultdict
# if __name__=='__main__':
#     t = int(input())
#     for i in range(t):
#         e,N = list(map(int, input().strip().split()))
#         arr = list(map(int, input().strip().split()))
#         graph = defaultdict(list)
#         creategraph(e, N, arr, graph)
#         res = topoSort(N, graph)
        
#         if check(graph, N, res):
#             print(1)
#         else:
#             print(0)


#            

# from collections import defaultdict

# def detectCycleUtil(adj, visited, v):
#     if visited[v] == 1:
#         return True
#     if visited[v] == 2:
#         return False
#     visited[v] = 1
#     for i in adj[v]:
#         if detectCycleUtil(adj, visited, i):
#             return True
#     visited[v] = 2
#     return False


# def detectCycle(adj, n):
#     visited = [0] * n
#     for i in range(n):
#         if visited[i] == False:
#             if detectCycleUtil(adj, visited, i):
#                 return True
#     return False

# class Graph():
#     def __init__(self,vertices):
#         self.graph = defaultdict(list)
#         self.V = vertices

#     def addEdge(self,u,v): # add directed edge from u to v.
#         self.graph[u].append(v)

# if __name__ == '__main__':
#     test_cases = int(input())
#     for cases in range(test_cases) :
#         N,E = map(int,input().strip().split()) 	#N=vertice,E=edges
#         g = Graph(N)
#         edges = list(map(int,input().strip().split()))

#         for i in range(0,len(edges),2):
#             u,v = edges[i],edges[i+1]
#             g.addEdge(u,v) # add a directed edge from u to v

#          # print bfs of graph
        
#         print(detectCycle(g.graph,N))



# import re
# month = []

# def updateLeapYear(year):
#     if year % 400 == 0:
#         month[2] = 29
#     elif year % 100 == 0:
#         month[2] = 28
#     elif year % 4 == 0:
#         month[2] = 29
#     else:
#         month[2] = 28

# def storeMonth():
#     month[1] = 31
#     month[2] = 28
#     month[3] = 31
#     month[4] = 30
#     month[5] = 31
#     month[6] = 30
#     month[7] = 31
#     month[8] = 31
#     month[9] = 30
#     month[10] = 31
#     month[11] = 30
#     month[12] = 31

# def findPrimeDates(d1, m1, y1, d2, m2, y2):
#     storeMonth()
#     result = 0

#     while(True):
#         x = d1
#         x = x * 100 + m1
#         x = x * 10000 + y1
#         if x % 4 == 0 or x % 7 == 0:
#             result = result + 1
#         if d1 == d2 and m1 == m2 and y1 == y2:
#             break
#         updateLeapYear(y1)
#         d1 = d1 + 1
#         if d1 > month[m1]:
#             m1 = m1 + 1
#             d1 = 1
#             if m1 > 12:
#                 y1 =  y1 + 1
#                 m1 =  1
#     return result;

# for i in range(1, 15):
#     month.append(31)

# line = input()
# date = re.split('-| ', line)
# d1 = int(date[0])
# m1 = int(date[1])
# y1 = int(date[2])
# d2 = int(date[3])
# m2 = int(date[4])
# y2 = int(date[5])

# result = findPrimeDates(d1, m1, y1, d2, m2, y2)
# print(result)

# 02-08-2025 04-09-2025

