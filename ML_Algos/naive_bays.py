# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1] # last column Y value
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

# # Test separating data by class
# dataset = [[3.393533211,2.331273381,0],
# 	[3.110073483,1.781539638,0],
# 	# [1.343808831,3.368360954,0],
# 	# [3.582294042,4.67917911,0],
# 	#[2.280362439,2.866990263,0],
# 	# [7.423436942,4.696522875,1],
# 	# [5.745051997,3.533989803,1],
# 	# [9.172168622,2.511101045,1],
# 	[7.792783481,3.424088941,1],
# 	[7.939820817,0.791637231,1]]
# separated = separate_by_class(dataset)
# for label in separated:
# 	print(label)
# 	for row in separated[label]:
# 		print(row)

# 0
# [3.393533211, 2.331273381, 0]
# [3.110073483, 1.781539638, 0]
# 1
# [7.792783481, 3.424088941, 1]
# [7.939820817, 0.791637231, 1]


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1]) # don't need summaries for Y column
	return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	#print(f"separated : {separated}")
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries


# # Test summarizing by class
# dataset = [[3.393533211,2.331273381,0],
# 	[3.110073483,1.781539638,0],
# 	#[1.343808831,3.368360954,0],
# 	#[3.582294042,4.67917911,0],
# 	#[2.280362439,2.866990263,0],
# 	#[7.423436942,4.696522875,1],
# 	#[5.745051997,3.533989803,1],
# 	[9.172168622,2.511101045,1],
# 	[7.792783481,3.424088941,1],
# 	[7.939820817,0.791637231,1]]
# summary = summarize_by_class(dataset)
# for label in summary:
# 	print(label)
# 	for row in summary[label]:
# 		print(row)


# separated : {0: [[3.393533211, 2.331273381, 0], 
# 				[3.110073483, 1.781539638, 0]], 
# 			 1: [[9.172168622, 2.511101045, 1], 
# 			     [7.792783481, 3.424088941, 1], 
# 			     [7.939820817, 0.791637231, 1]]}
# 0
# (3.251803347, 0.20043629586209424, 2)
# (2.0564065095, 0.38872045752236273, 2)
# 1
# (8.301590973333333, 0.7575183669806166, 3)
# (2.242275739, 1.3366565696655128, 3)


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	#print(f"total_rows : {total_rows}")
	#print(f"summaries : {summaries}")
	#print(f"summaries[label][0][2] : {[summaries[label][0][2] for label in summaries]}")
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		#print(f"probabilities : {probabilities}") #[class=0,class=1]
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			#print(f"row[{i}] : {row[i]} , class_summaries[{i}] : {class_summaries[i]}")
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities

# # Test calculating class probabilities
# dataset = [[3.393533211,2.331273381,0],
# 	[3.110073483,1.781539638,0],
# 	# [1.343808831,3.368360954,0],
# 	# [3.582294042,4.67917911,0],
# 	# [2.280362439,2.866990263,0],
# 	# [7.423436942,4.696522875,1],
# 	# [5.745051997,3.533989803,1],
# 	# [9.172168622,2.511101045,1],
# 	[7.792783481,3.424088941,1],
# 	[7.939820817,0.791637231,1]]
# summaries = summarize_by_class(dataset)
# probabilities = calculate_class_probabilities(summaries, dataset[0])
# print(f"final probabilities : {probabilities}")

# total_rows : 4
# summaries : {0: [(3.251803347, 0.20043629586209424, 2), 
# 				(2.0564065095, 0.38872045752236273, 2)], 
# 			 1: [(7.866302149, 0.10397109737320512, 2), 
# 			      (2.107863086, 1.8614244552871226, 2)]}
# summaries[label][0][2] : [2, 2] # [label=0 , label=1]
# probabilities : {0: 0.5}
# row[0] : 3.393533211 , class_summaries[0] : (3.251803347, 0.20043629586209424, 2)
# row[1] : 2.331273381 , class_summaries[1] : (2.0564065095, 0.38872045752236273, 2)
# probabilities : {0: 0.6194826244620996, 1: 0.5}
# row[0] : 3.393533211 , class_summaries[0] : (7.866302149, 0.10397109737320512, 2)
# row[1] : 2.331273381 , class_summaries[1] : (2.107863086, 1.8614244552871226, 2)
# final probabilities : {0: 0.6194826244620996, 1: 0.0}


# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

# Make a prediction with Naive Bayes on Iris Dataset
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# fit model
model = summarize_by_class(dataset)
# define a new record
row = [5.7,2.9,4.2,1.3]
# predict the label
label = predict(model, row)
print('Data=%s, Predicted: %s' % (row, label))