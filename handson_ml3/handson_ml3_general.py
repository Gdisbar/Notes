skfolds = StratifiedKFold(n_splits=3)
for train_index,test_index in skfolds.split(X_train,y_train):
	clf = clone(orginal_clf)
	X_train_fold = X_train[train_index]
	y_train_fold = y_train[train_index]
	X_test_fold = X_test[test_index]
	y_test_fold = y_test[test_index]

	clf.fit(X_train_fold,y_train_fold)
	y_pred = clf.predict(X_test_fold)
	n_correct = sum(y_pred==y_test_fold)
	print(n_correct/len(y_pred))

plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
--------------------------------------------------------------------------------
## image augmentation
---------------------------------------------------------------------------------
from scipy.ndimage import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


X_train_augmented = [image for image in X_train] ## orginal images
y_train_augmented = [label for label in y_train] ## orginal labels

## append shifted image along with orginal one

for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

## tuned_accuracy = grid_search.score(X_test, y_test)
## knn_clf = KNeighborsClassifier(**grid_search.best_params_)
## augmented_accuracy = knn_clf.score(X_test, y_test)
## error_rate_change = (1 - augmented_accuracy) / (1 - tuned_accuracy) - 1

### reset the name counters and make the code reproducible
tf.keras.backend.clear_session()
tf.random.set_seed(42)

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]

for batch in mnist_train.shuffle(10_000, seed=42).batch(32).prefetch(1):
    images = batch["image"]
    labels = batch["label"]
    # [...] do something with the images and labels

mnist_train = mnist_train.shuffle(10_000, seed=42).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefetch(1)

