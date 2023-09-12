==========================================================================
==========================================================================
---------------------------------------------------------------------------
########## Classification ##########################
---------------------------------------------------------------------------
==========================================================================
==========================================================================

## X_train -> (2400,) , y_train -> ()

## basic metrics
---------------------------------------------------------------------------
#sklearn.model_selection
cv_score = cross_val_score(clf,y_val,y_val_pred,cv=3)
## predict ~ true
## TN FP
## FN TP
#sklearn.metrics
cm = confusion_matrix(y_test,y_pred) 
## precision = TP / (FP+TP) 
## recall = TP / (FN+TP)
precision = precision_score(y_test,y_pred) = cm[1,1]/(cm[0,1]+cm[1,1])
f1_score = f1_score(y_test,y_pred)



### getting precision-recall at some threshold
--------------------------------------------------------------------------
threshold = 80 # at 80% threshold we want to know the precision,recall value
y_score = cross_val_predict(clf,X_train,y_train,cv=3,method="decision_function")
precisions,recalls,thresholds = precision_recall_curve(y_train,y_score)

## plotting -> threshold 
plt.plot(thresholds,precisions[:-1]) ## same for recall
plt.vline(threshold,y_min,y_max) ## y_min,y_max -> range at which line drawn
idx = (thresholds>=threshold).argmax() ## get index for max precision,recall value
plt.plot(thresholds[idx],precisions[idx]) ## same recall
plt.axis([x_min,x_max,y_min,y_max])
plt.grid() ## use to calculate distance
plt.text(x,y,"precision_recall_curve") # x,y -> positions

## plotting -> recall ~ precision , shape is similar to roc curve 
plt.plot([recalls[idx],recalls[idx]], [0,precisions[idx]])
plt.plot([0,recalls[idx]],[precisions[idx],precisions[idx]])
plt.plot([recalls[idx]],[precisions[idx]],label="Point at threshold 3,000")

## get precisoion indx -> find threshold at that index -> use that threshold to get
## cross validation predict (y_score) -> use that y_score to get new recall
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision] # 3370.0194991439557
y_train_pred_90 = (y_scores >= threshold_for_90_precision)
#precision_at_90_precision=precision_score(y_train, y_train_pred_90) # 0.9000345901072293
recall_at_90_precision = recall_score(y_train, y_train_pred_90) # 0.4799852425751706


### roc curve - fpr(fall-out) ~ tpr(recall)
-------------------------------------------------------------------------------------
fpr,trp,thresholds = roc_curve(y_train,y_train_pred_90)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.plot(fpr,tpr) ## roc curve
plt.plot([0, 1],[0, 1]) # worst case
plt.plot([fpr_90],[tpr_90]) # threshold for 90% precision


## for randomforest -> y_probas_forest = cross_val_prredict(method="predict_proba")

## how many images(94%) has been classified +ve with prob of 50%~60% ?
idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum() # 94%
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train, y_scores_forest)

=================================================================================
### Multiclass classification (sklearn.multiclass)
=================================================================================
## SVC doesn't scale to large data 
## SVC uses OvO during training but if we want to use OvR
ovr_clf = OneVsRestClassifier(SVC(random_state=42))
# ovr_clf.predict([X[0]])
# ovr_clf.estimators_   # 10
## another way of using OvO
svm_clf.decision_function_shape ="ovo"
score_ovo = svm_clf.decision_function([X[0]]).round(2)

=================================================================================
### Error Analysis
=================================================================================
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
sample_weight = (y_train_pred != y_train) # without this only diagonal is selected
plt.rc('font', size=10)  
# predicted label ~ true label - Error normalized by Row
# use normalize="pred" to normalized by Column
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")

## now display the misclassified labels - X_ba => X_train==b and X_train_pred==a
size = 5
pad = 0.2
for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
                                       (X_aa, (0, 1)), (X_ab, (1, 1))]:
    for idx, image_data in enumerate(images[:size*size]):
        x = idx % size + label_col * (size + pad)
        y = idx // size + label_row * (size + pad)
        plt.imshow(image_data.reshape(28, 28), cmap="binary",
                   extent=(x, x + 1, y, y + 1))
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])


=================================================================================
### Multilabel classification 
=================================================================================


## target label - combination of labels associated with a single target
# y_train_large = (y_train >= '7')
# y_train_odd = (y_train.astype('int8') % 2 == 1)
# y_multilabel = np.c_[y_train_large, y_train_odd]

# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

Inherently multiclass: 

NB,DT,KNN,Linear_SVC(multi_class=”crammer_singer”),LDA,
LogisticRegressionCV(multi_class=”multinomial”)


------------------------------------------------------------------------------
## 1. OneVsRest (problem transformations)
------------------------------------------------------------------------------
# build multiple independent classifiers and, 
# for an unseen instance, choose the class for which the confidence is maximized.
# The main assumption here is that the labels are mutually exclusive

## questions like -> is the comment toxic or not?,is the comment threatening or not?

ovr_clf = OneVsRestClassifier(LogisticRegression(solver='sag'))
LogReg_pipeline = Pipeline([('clf',ovr_clf)])
categories = list(data_raw.columns.values)

for category in categories:
	LogReg_pipeline.fit(x_train, train[category])
	prediction = LogReg_pipeline.predict(x_test) # test accuracy
	#accuracy_score(test[category], prediction)

------------------------------------------------------------------------------
## 2.Binary Relevance
------------------------------------------------------------------------------

# In this case an ensemble of single-label binary classifiers is trained, 
# one for each class. Each classifier predicts either the membership or 
# the non-membership of one class. The union of all classes that were predicted 
# is taken as the multi-label output. It ignores the possible correlations 
# between class labels.

# if there’s q labels, the binary relevance method create q new data sets 
# from the images, one for each label and train single-label classifiers on each 
# new data set. One classifier may answer yes/no to the question 
# “does it contain trees?”, thus the “binary” in “binary relevance”. This is a 
# simple approach but does not work well when there’s dependencies between the labels.

# OneVsRest & Binary Relevance seem very much alike. If multiple classifiers in 
# OneVsRest answer “yes” then you are back to the binary relevance scenario.

from skmultilearn.problem_transform import BinaryRelevance
classifier = BinaryRelevance(GaussianNB())
## fit() + predict() + accracy_score()

------------------------------------------------------------------------------
## 3. Classifier Chains
------------------------------------------------------------------------------
# A chain of binary classifiers C0, C1, . . . , Cn is constructed, 
# (in a serial manner from C2 data = train set + C1)where a classifier Ci uses 
# the predictions of all the classifier Cj , where j < i. Correlations 
# between class labels is addessed here

from skmultilearn.problem_transform import ClassifierChain

classifier = ClassifierChain(LogisticRegression())
## fit() + predict() + accracy_score()

------------------------------------------------------------------------------
## 4. Label Powerset (algorithm adaptation)
------------------------------------------------------------------------------
# This approach does take possible correlations between class labels into account. 
# More commonly this approach is called the label-powerset method, because it 
# considers each member of the power set of labels in the training set as a single 
# label. This method needs worst case (2^|C|) classifiers

from skmultilearn.problem_transform import LabelPowerset
classifier = LabelPowerset(LogisticRegression())

------------------------------------------------------------------------------
### 5. Adapted Algorithm
------------------------------------------------------------------------------
# Algorithm adaptation methods for multi-label classification concentrate on 
# adapting single-label classification algorithms to the multi-label case usually 
# by changes in cost/decision functions.
# Here we use a multi-label lazy learning approach named ML-KNN which is derived 
# from the traditional K-nearest neighbor (KNN) algorithm.


from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
classifier_new = MLkNN(k=10)
# Note that this classifier can throw up errors when handling sparse matrices.
x_train = lil_matrix(x_train).toarray()
y_train = lil_matrix(y_train).toarray()
x_test = lil_matrix(x_test).toarray()


# The same problem can be solved using LSTMs in deep learning.

# For more speed we could use decision trees and for a reasonable trade-off 
# between speed and accuracy we could also opt for ensemble models.

# Other frameworks such as MEKA can be used to deal with multi-label 
# classification problems.

=================================================================================
### Multioutput classification (sklearn.multioutput)
=================================================================================

# chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
# chain_clf.fit(X_train[:2000], y_multilabel[:2000])
# chain_clf.predict([X[0]]) # array([[0., 1.]]) 

# noise = np.random.randint(0, 100, (len(X_train), 784))
# X_train_mod = X_train + noise
# noise = np.random.randint(0, 100, (len(X_test), 784))
# X_test_mod = X_test + noise
# y_train_mod = X_train
# y_test_mod = X_test

# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[0]])

Support multiclass-multioutput:

    tree.DecisionTreeClassifier
    tree.ExtraTreeClassifier
    ensemble.ExtraTreesClassifier
    neighbors.KNeighborsClassifier
    neighbors.RadiusNeighborsClassifier
    ensemble.RandomForestClassifier

X, y1 = make_classification(n_samples=10, n_features=100,
	                            n_informative=30, n_classes=3,
	                            random_state=1)
y2 = shuffle(y1, random_state=1)
y3 = shuffle(y1, random_state=2)
Y = np.vstack((y1, y2, y3)).T
n_samples, n_features = X.shape # 10,100
n_outputs = Y.shape[1] # 3
n_classes = 3
forest = RandomForestClassifier(random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=2)
multi_target_forest.fit(X, Y).predict(X)
# array([[2, 2, 0],
#        [1, 2, 1],
#        [2, 1, 0],
#        [0, 0, 2],
#        [0, 2, 1],
#        [0, 0, 2],
#        [1, 1, 0],
#        [1, 1, 1],
#        [0, 0, 2],
#        [2, 0, 0]])

# A column wise concatenation of 1d multiclass variables. 
# (n_samples, n_classes)
y = np.array([['apple', 'green'], ['orange', 'orange'], ['pear', 'green']])
# [['apple' 'green']
#  ['orange' 'orange']
#  ['pear' 'green']]

y = np.array([[31.4, 94], [40.5, 109], [25.0, 30]]) # (n_samples, n_output) 
# [[ 31.4  94. ]
#  [ 40.5 109. ]
#  [ 25.   30. ]]

multioutput_reg = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))

# category -> [Pharmacutical,HealthCare],[Clinical Research],[Manufacturing,Pharmacutical,HealthCare,].....
X = df.loc[:,"job_description"]
y= df[["job_type","category"]] 
pipe = Pipeline([("c_v",cv),("lr",MultiOutputClassifier(lr))])


# we will use the yeast dataset which contains 2417 datapoints each with
# 103 features and 14 possible labels. Each data point has at least one label. 
# As a baseline we first train a logistic regression classifier for each of the 14 
# labels. To evaluate the performance of these classifiers we predict on a held-out 
# test set and calculate the jaccard score for each sample.


# Next we create 10 classifier chains. Each classifier chain contains a logistic 
# regression model for each of the 14 labels. The models in each chain are ordered 
# randomly. In addition to the 103 features in the dataset, each model gets the 
# predictions of the preceding models in the chain as features (note that by default 
# at training time each model gets the true labels as features). These additional 
# features allow each chain to exploit correlations among the classes. The Jaccard 
# similarity score for each chain tends to be greater than that of the set independent 
# logistic models.

# Because the models in each chain are arranged randomly there is significant 
# variation in performance among the chains. Presumably there is an optimal 
# ordering of the classes in a chain that will yield the best performance. However 
# we do not know that ordering a priori. Instead we can construct an voting ensemble 
# of classifier chains by averaging the binary predictions of the chains and apply 
# a threshold of 0.5. The Jaccard similarity score of the ensemble is greater than 
# that of the independent models and tends to exceed the score of each chain in 
# the ensemble (although this is not guaranteed with randomly ordered chains).


# X, Y = sklearn.datasets.fetch_openml("yeast", version=4, return_X_y=True, 
# 										parser="pandas")
Y = Y == "TRUE"

# Fit an independent logistic regression model for each class using the
# OneVsRestClassifier wrapper.
base_lr = LogisticRegression()
ovr = OneVsRestClassifier(base_lr)
ovr.fit(X_train, Y_train)
Y_pred_ovr = ovr.predict(X_test)
ovr_jaccard_score = jaccard_score(Y_test, Y_pred_ovr, average="samples")


# Fit an ensemble of logistic regression classifier chains and take the
# average prediction of all the chains.
chains = [ClassifierChain(base_lr, order="random", random_state=i) for i in range(10)]
for chain in chains:
    chain.fit(X_train, Y_train)

Y_pred_chains = np.array([chain.predict(X_test) for chain in chains])
chain_jaccard_scores = [
    jaccard_score(Y_test, Y_pred_chain >= 0.5, average="samples")
    for Y_pred_chain in Y_pred_chains
]

Y_pred_ensemble = Y_pred_chains.mean(axis=0)
ensemble_jaccard_score = jaccard_score(
    Y_test, Y_pred_ensemble >= 0.5, average="samples"
)

model_scores = [ovr_jaccard_score] + chain_jaccard_scores
model_scores.append(ensemble_jaccard_score)

model_names = (
    "Independent",
    "Chain 1",
    "Chain 2",
    "Chain 3",
    "Chain 4",
    "Chain 5",
    "Chain 6",
    "Chain 7",
    "Chain 8",
    "Chain 9",
    "Chain 10",
    "Ensemble",
)

x_pos = np.arange(len(model_names))

# Plot the Jaccard similarity scores for the independent model, each of the
# chains, and the ensemble (note that the vertical axis on this plot does
# not begin at 0).

ax.set_title("Classifier Chain Ensemble Performance Comparison")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation="vertical")
ax.set_ylabel("Jaccard Similarity Score")
ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
