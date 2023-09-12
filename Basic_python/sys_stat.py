import re,os,time
from itertools import islice
import pandas as pd 

r_n = 11  ## number of records 
n_t = 10  ## number of times we want to generate the data
n = r_n*n_t + r_n  ## 110 is the last line with data(just previous to the cursor)

cmd1 = 'cat /proc/cpuinfo | grep "cache size" | uniq  >> op.txt'
cmd2 = 'cat /proc/cpuinfo | grep "cpu MHz" | uniq >> op.txt'
cmd3 = 'cat /proc/meminfo | grep "Buffers" |uniq >> op.txt'
cmd4 = 'cat /proc/meminfo | grep "Cached" |uniq >> op.txt'
cmd5 = 'cat /proc/meminfo | grep "SwapFree" |uniq >> op.txt'
cmd6 = 'cat /proc/meminfo | grep "MemFree" |uniq >> op.txt'
cmd7 = 'cat /proc/meminfo | grep "Dirty" |uniq >> op.txt'

for i in range(n_t):
	os.system(cmd1)
	os.system(cmd2)
	os.system(cmd3)
	os.system(cmd4)
	os.system(cmd5)
	os.system(cmd6)
	os.system(cmd7)
	time.sleep(1)

print("Raw data saved into op.txt")
with open('op.txt') as f:
	lis = []
	for i in range(n):
		text = f.readline()
		x = re.sub("[a-zA-Z:\t\n]","",text)
		lis.append(str(x).strip())

#print(lis)

lis2 = []
for _ in range(r_n):
	lis2.append(r_n)

def lis_convert(lis, lis2): 
    it = iter(lis) 
    return [list(islice(it, i)) for i in lis2] 

lis3 = lis_convert(lis, lis2)

#print(lis3)

df = pd.DataFrame(lis3, columns =['cache size','cpu(core 1) MHz','cpu(core 2) MHz','cpu(core 3) MHz','cpu(core 4) MHz',
	'Buffers','Cached','SwapCached','SwapFree','MemFree','Dirty'], dtype = float) 

#print(df) 

df.to_csv('op.csv')
print("Formatted data saved into op.csv")
#df = pd.read_csv('op.csv',index_col=0)