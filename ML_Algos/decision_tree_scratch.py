# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		# # here Y has two classes [0,1]
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			print(f"class_value : {class_val}")
			# print(f"[row[-1] for row in group].count(class_val) {p}")
			print(f"p : {p}")
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# # test Gini values
# print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))

# [[[1, 1], [1, 0]], [[1, 1], [1, 0]]]
# group : [[1, 1], [1, 0]]
# class_val : 0 
# row[-1] : [1, 0]
# p : 1
# class_val : 1 
# row[-1] : [1, 0]
# p : 1
# group : [[1, 1], [1, 0]]
# class_val : 0 
# row[-1] : [1, 0]
# p : 1
# class_val : 1 
# row[-1] : [1, 0]
# p : 1

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			print(f"row : {row} , row[{index}] : {row[index]}")
			print(f"groups : {groups}")
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	print(f"outcomes : {outcomes}")
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	print(f"left : {left} , right : {right}")
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	#root = {'index':b_index, 'value':b_value, 'groups':b_groups}
	root = get_split(train) 
	split(root, max_depth, min_size, 1)
	return root

# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))


# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

dataset = [[2.771244718,1.784783929,0],
	[1.728571309,1.169761413,0],
	# [3.678319846,2.81281357,0],
	# [3.961043357,2.61995032,0],
	# [2.999208922,2.209014212,0],
	# [7.497545867,3.162953546,1],
	# [9.00220326,3.339047188,1],
	# [7.444542326,0.476683375,1],
	# [10.12493903,3.234550982,1],
	[6.642287351,3.319983761,1]]
tree = build_tree(dataset, 1, 1)
print_tree(tree)

#  predict with a stump
stump = {'index': 0, 'right': 1, 'value': 6.642287351, 'left': 0}
for row in dataset:
	prediction = predict(stump, row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

# row : [2.771244718, 1.784783929, 0] , row[0] : 2.771244718
# groups : ([[1.728571309, 1.169761413, 0]], 
# 	      [[2.771244718, 1.784783929, 0], [6.642287351, 3.319983761, 1]])
# class_value : 0
# p : 1.0
# class_value : 1
# p : 0.0
# class_value : 0
# p : 0.5
# class_value : 1
# p : 0.5
# row : [1.728571309, 1.169761413, 0] , row[0] : 1.728571309
# groups : ([],[[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0], 
# 	[6.642287351, 3.319983761, 1]])
# class_value : 0
# p : 0.6666666666666666
# class_value : 1
# p : 0.3333333333333333
# row : [6.642287351, 3.319983761, 1] , row[0] : 6.642287351
# groups : ([[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0]], 
# 			[[6.642287351, 3.319983761, 1]])
# class_value : 0
# p : 1.0
# class_value : 1
# p : 0.0
# class_value : 0
# p : 0.0
# class_value : 1
# p : 1.0
# row : [2.771244718, 1.784783929, 0] , row[1] : 1.784783929
# groups : ([[1.728571309, 1.169761413, 0]], 
# 			[[2.771244718, 1.784783929, 0], [6.642287351, 3.319983761, 1]])
# class_value : 0
# p : 1.0
# class_value : 1
# p : 0.0
# class_value : 0
# p : 0.5
# class_value : 1
# p : 0.5
# row : [1.728571309, 1.169761413, 0] , row[1] : 1.169761413
# groups : ([], [[2.771244718, 1.784783929, 0], 
# 	        [1.728571309, 1.169761413, 0], [6.642287351, 3.319983761, 1]])
# class_value : 0
# p : 0.6666666666666666
# class_value : 1
# p : 0.3333333333333333
# row : [6.642287351, 3.319983761, 1] , row[1] : 3.319983761
# groups : ([[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0]], 
# 	      [[6.642287351, 3.319983761, 1]])
# class_value : 0
# p : 1.0
# class_value : 1
# p : 0.0
# class_value : 0
# p : 0.0
# class_value : 1
# p : 1.0
# left : [[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0]] , 
# right : [[6.642287351, 3.319983761, 1]]
# outcomes : [0, 0]
# outcomes : [1]



# [X1 < 6.642]
#  [0]
#  [1]


# Expected=0, Got=0
# Expected=0, Got=0
# Expected=1, Got=1



seed(1)
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
n_folds = 5
max_depth = 5
min_size = 10
#algorithm = "decision_tree"
# Evaluate an algorithm using a cross validation split
scores = evaluate_algorithm(dataset,decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

Scores: [96.35036496350365, 97.08029197080292, 97.44525547445255, 
			98.17518248175182, 97.44525547445255] # n_folds = 5
Mean Accuracy: 97.299%