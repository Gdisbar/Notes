# OS module
#
#
import os
# Available methods
#print(dir(os))
# Access Path/Files
print(os.getcwd())
#os.chdir('/home/acroo/Programs/')
## running commands from python
import subprocess
command = os.system("pwd")      #os.system('command') for single command on currect working directory
command2 = subprocess.call(['/bin/ls','-laS','/home/acroo/Music/'])  #subprocess.call('['/bin/ls','-laS','/home/acroo/Music/']') for 
                                                                        #single command on any directory
command3 = os.popen("pwd").read()       #os.popen('command').read() to store command
print(command)
print(command2)
print(command3)

# List File.Folders
#print(os.listdir())
import glob
for file_name in glob.iglob('/home/acroo/Programs/Lang_Python/**/*.pdf',recursive=True):
	print(file_name)

print("#################################")	
for name in glob.glob('/home/acroo/Programs/Lang_Python/*[0-9].*'): 
    print(name) 
    
print("#################################")    
char_seq = "-_#"
  
for spcl_char in char_seq: 
    esc_set = "*" + glob.escape(spcl_char) + "*" + ".py"
      
    for x in (glob.glob(esc_set)): 
        print(x) 

print("#################################")
def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        # print(dirs)
        if dir!= '.git':
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
startpath = os.getcwd()
list_files(startpath)
print("#################################")

# Create Single/Nested Directory
#os.mkdir('testdir')
#os.makedirs('level1dir/level2dir')
#os.rmdir('testdir')
#os.removedirs('level1dir/level2dir')

# Data Pre-Processing
import pandas as pd

# create a list to hold the data from each state
list_states = []

# iteratively loop over all the folders and add their data to the list
for root, dirs, files in os.walk(os.getcwd()):
    if files:
        list_states.append(pd.read_csv(root+'/'+files[0], index_col=None))

# merge the dataframes into a single dataframe using Pandas library
merge_data = pd.concat(list_states[1:], sort=False)




