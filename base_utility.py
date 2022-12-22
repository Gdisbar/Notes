
# concat multiple test file
------------------------------

# Load all txt files in path
files = glob.glob('/path/to/files/*.txt')

# Concatenate files to new file
with open('2020_output.txt', 'w') as out_file:
    for file_name in files:
        with open(file_name) as in_file:
            out_file.write(in_file.read())

# Read file and print
with open('2020_output.txt', 'r') as new_file:
    lines = [line.strip() for line in new_file]
for line in lines: print(line)


# Concatenate Multiple CSV Files Into a DataFrame
------------------------------------------------------

# Load all csv files in path
files = glob.glob('/path/to/files/*.csv')

# Create a list of dataframe, one series per CSV
fruit_list = []
for file_name in files:
    df = pd.read_csv(file_name, index_col=None, header=None)
    fruit_list.append(df)

# Create combined frame out of list of individual frames
fruit_frame = pd.concat(fruit_list, axis=0, ignore_index=True)

print(fruit_frame)

# sort list of tuple



# Some paired data
pairs = [(1, 10.5), (5, 7.), (2, 12.7), (3, 9.2), (7, 11.6)]

# Sort pairs by first entry
sorted_pairs  = sorted(pairs, key=lambda x: x[0])
print(f'Sorted by element 0 (first element):\n{sorted_pairs}')

# Sort pairs by second entry
sorted_pairs  = sorted(pairs, key=lambda x: x[1])
print(f'Sorted by element 1 (second element):\n{sorted_pairs}')

# Extend this to tuples of size n and non-numeric entries
pairs = [('banana', 3), ('apple', 11), ('pear', 1), ('watermelon', 4), ('strawberry', 2), ('kiwi', 12)]
sorted_pairs  = sorted(pairs, key=lambda x: x[0])
print(f'Alphanumeric pairs sorted by element 0 (first element):\n{sorted_pairs}')



