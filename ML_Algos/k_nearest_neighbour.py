# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	#print(f"distances : {distances}")
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	#for neighbor in neighbors:
	#print(f"neighbors : {neighbors}")
	output_values = [row[-1] for row in neighbors] # we only need Y=predicted class
	#print(f"output_values : {output_values}")
	prediction = max(set(output_values), key=output_values.count)
	return prediction

# # Test distance function
# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	# [3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]

# prediction = predict_classification(dataset, dataset[0], 3)
# print('Expected %d, Got %d.' % (dataset[0][-1], prediction))


distances : [([2.7810836, 2.550537003, 0], 0.0), 
             ([1.465489372, 2.362125076, 0], 1.3290173915275787), 
             ([1.38807019, 1.850220317, 0], 1.5591439385540549), 
             ([3.396561688, 4.400293529, 0], 1.9494646655653247), 
             ([5.332441248, 2.088626775, 1], 2.592833759950511), 
             ([6.922596716, 1.77106367, 1], 4.214227042632867), 
             ([7.627531214, 2.759262235, 1], 4.850940186986411), 
             ([7.673756466, 3.508563011, 1], 4.985585382449795), 
             ([8.675418651, -0.242068655, 1], 6.522409988228337)]
neighbors : [[2.7810836, 2.550537003, 0], 
             [1.465489372, 2.362125076, 0], 
             [1.38807019, 1.850220317, 0]]
output_values : [0, 0, 0]
Expected 0, Got 0.

# Make a prediction with KNN on Iris Dataset
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# define model parameter
num_neighbors = 5
# define a new record
row = [5.7,2.9,4.2,1.3]
# predict the label
label = predict_classification(dataset, row, num_neighbors)
print('Data=%s, Predicted: %s' % (row, label))