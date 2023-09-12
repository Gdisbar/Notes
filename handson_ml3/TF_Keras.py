#plot model
tf.keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
# extra code – shows how to convert class ids to one-hot vectors
tf.keras.utils.to_categorical([0, 5, 1, 0], num_classes=10)

# extra code – shows how to shift the training curve by -1/2 epoch
-------------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
for key, style in zip(history.history, ["r--", "r--.", "b-", "b-*"]):
    epochs = np.array(history.epoch) + (0 if key.startswith("val_") else -0.5)
    plt.plot(epochs, history.history[key], style, label=key)
plt.xlabel("Epoch")
plt.axis([-0.5, 29, 0., 1])
plt.legend(loc="lower left")
plt.grid()
plt.show()

# axis=-1 in np.argmax() is similar to list from last

# extra code – split Fashion MNIST into tasks A and B, then train and save
#              model A to "my_model_A".

# pos_class_id = class_names.index("Pullover")
# neg_class_id = class_names.index("T-shirt/top")

# def split_dataset(X, y):
#     y_for_B = (y == pos_class_id) | (y == neg_class_id)
#     y_A = y[~y_for_B] # all except pos_class_id & neg_class_id
#     y_B = (y[y_for_B] == pos_class_id).astype(np.float32) # only pos_class_id 
#     old_class_ids = list(set(range(10)) - set([neg_class_id, pos_class_id])) # 8 class
#     for old_class_id, new_class_id in zip(old_class_ids, range(8)):
#         y_A[y_A == old_class_id] = new_class_id  # reorder class ids for A
#     return ((X[~y_for_B], y_A), (X[y_for_B], y_B))

# (X_train_A, y_train_A) --> all except ["Pullover":2,"T-shirt/top":10] 
# (X_train_B, y_train_B) --> only "Pullover" or both ["Pullover":2,"T-shirt/top":10] 
# y_B [0. 0. 0. ... 1. 0. 1.]
# y_for_B [False  True  True ... False  True  True]
# plt.imshow(X[y_for_B[1]][0][-2]) -> 1 & -2 are same, i.e taking both
# old_class_ids [1, 3, 4, 5, 6, 7, 8, 9]
# y_A [7 1 5 ... 7 0 7]

======================================================================================
## Functional API - sending different inputs
----------------------------------------------------------------------------------
input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])


# Subclassing API to Build Dynamic Models
-------------------------------------------------------------------------------------
class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)  # needed to support naming the model
        self.norm_layer_wide = tf.keras.layers.Normalization()
        self.norm_layer_deep = tf.keras.layers.Normalization()
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.hidden2 = tf.keras.layers.Dense(units, activation=activation)
        self.main_output = tf.keras.layers.Dense(1)
        self.aux_output = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        input_wide, input_deep = inputs
        norm_wide = self.norm_layer_wide(input_wide)
        norm_deep = self.norm_layer_deep(input_deep)
        hidden1 = self.hidden1(norm_deep)
        hidden2 = self.hidden2(hidden1)
        concat = tf.keras.layers.concatenate([norm_wide, hidden2])
        output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return output, aux_output

model = WideAndDeepModel(30, activation="relu", name="my_cool_model")
model.compile(loss="mse", loss_weights=[0.9, 0.1], optimizer=optimizer,
              metrics=["RootMeanSquaredError"])
# direct calculation is possible - due to BatchNormalization() as it store mean & std
model.norm_layer_wide.adapt(X_train_wide)
model.norm_layer_deep.adapt(X_train_deep)


history = model.fit((X_train_wide, X_train_deep), (y_train, y_train), epochs=10,
    validation_data=((X_valid_wide, X_valid_deep), (y_valid, y_valid)))
eval_results = model.evaluate((X_test_wide, X_test_deep), (y_test, y_test))
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results
y_pred_main, y_pred_aux = model.predict((X_new_wide, X_new_deep))


=========================================================================================
## Transfer Learning

# split Fashion MNIST into tasks A and B, then train and save
# model A to "my_model_A".

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

pos_class_id = class_names.index("Pullover")
neg_class_id = class_names.index ("T-shirt/top")

## X_A,y_A = all except pos & neg
## X_B,y_B = only pos
def split_dataset(X, y):
    y_for_B = (y == pos_class_id) | (y == neg_class_id)
    y_A = y[~y_for_B]
    y_B = (y[y_for_B] == pos_class_id).astype(np.float32)
    old_class_ids = list(set(range(10)) - set([neg_class_id, pos_class_id]))
    for old_class_id, new_class_id in zip(old_class_ids, range(8)):
        y_A[y_A == old_class_id] = new_class_id  # reorder class ids for A
    return ((X[~y_for_B], y_A), (X[y_for_B], y_B))

(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

# model_A -> (X_train_A, y_train_A) & (X_valid_A, y_valid_A) | (X_test_A, y_test_A)
# model_B -> (X_train_B, y_train_B) & (X_valid_B, y_valid_B) | (X_test_B, y_test_B)

model_A = tf.keras.models.load_model("model_A")
# when we train one, it will update both models. If we want to avoid that, 
# we need to build model_B_on_A on top of a clone of model_A

model_A_clone = tf.keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
# extra code – creating model_B_on_A just like in the previous cell
model_B_on_A = tf.keras.Sequential(model_A_clone.layers[:-1])
model_B_on_A.add(tf.keras.layers.Dense(1, activation="sigmoid"))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

# model_B_on_A --> (X_train_B, y_train_B) & (X_valid_B, y_valid_B) | (X_test_B, y_test_B)

for layer in model_B_on_A.layers[:-1]: # now training the complete model
    layer.trainable = True

# [0.2546142041683197, 0.9384999871253967]
#1 - (100 - model_B_on_A ) / (100 - model_B) --> error rate dropped by 25%

# some optimizer settings
--------------------------------
SGD(learning_rate=0.001, momentum=0.9,nesterov=True)
Adagrad(learning_rate=0.001)
RMSprop(learning_rate=0.001, rho=0.9)
Adam(learning_rate=0.001, beta_1=0.9,beta_2=0.999)
# Adamax(learning_rate=0.001, beta_1=0.9,beta_2=0.999)
# Nadam(learning_rate=0.001, beta_1=0.9,beta_2=0.999)
# from tensorflow_addons.optimizers import AdamW
# AdamW(weight_decay=1e-5, learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# Regularization
===================================================================================
# Or use l1(0.1) for ℓ1 regularization with a factor of 0.1, 
# or l1_l2(0.1, 0.01) for both ℓ1 and ℓ2 regularization, with factors 0.1 
# and 0.01 respectively.

from functools import partial
# Partial functions allow us to fix a certain number of arguments 
# of a function and generate a new function.

# def add(a, b, c): return 100 * a + 10 * b + c
# g = partial(add, c = 2, b = 1)
# print(add_part(3)) -----------> 312

### Another example
# ------------------------------------
# def f(a, b, c, x): return 1000*a + 100*b + 10*c + x
# g = partial(f, 3, 1, 4)
# print(g(5)) -------> 3145

RegularizedDense = partial(tf.keras.layers.Dense,
                           activation="relu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(100),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])

===================================================================================
# MC Dropout
---------------------------------------------------------------------------------
# with normal dropout , accuracy = [0.3603347837924957, 0.8711000084877014] 
model.evaluate(X_test, y_test) 
# adding MC Dropout -> y_probas= (100, 10000, 10)
# where X_test = (10000, 28, 28), y_test = (10000,) 
# here no need to fit once mode, just use training=True during prediction
y_probas = np.stack([model(X_test, training=True) for sample in range(100)])
y_proba = y_probas.mean(axis=0)
# model.predict(X_test[:1]).round(3) # 10 classes
# y_std = y_probas.std(axis=0)
y_pred = y_proba.argmax(axis=1)
accuracy = (y_pred == y_test).sum() / len(y_test) # 87%

### Alternate
-----------------------------------------------------------------------------------
# we add MC Dropout to pre-build model , we re-train with MC Dropout
# or we can just use model(X_test, training=True)
class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

Dropout = tf.keras.layers.Dropout
mc_model = tf.keras.Sequential([
    MCDropout(layer.rate) if isinstance(layer, Dropout) else layer
    for layer in model.layers
])
mc_model.set_weights(model.get_weights())


np.mean([mc_model.predict(X_test[:1])
         for sample in range(100)], axis=0).round(2) # 87%
------------------------------------------------------------------------------------
#Max norm
----------------------------------------------
dense = tf.keras.layers.Dense(
    100, activation="relu", kernel_initializer="he_normal",
    kernel_constraint=tf.keras.constraints.max_norm(1.))
# extra code – shows how to apply max norm to every hidden layer in a model

MaxNormDense = partial(tf.keras.layers.Dense,
                       activation="relu", kernel_initializer="he_normal",
                       kernel_constraint=tf.keras.constraints.max_norm(1.))

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    MaxNormDense(100),
    MaxNormDense(100),
    tf.keras.layers.Dense(10, activation="softmax")
])

====================================================================================
# Learning Rate & Custom Callbacks
----------------------------------------

#PrintValTrainRatioCallback
--------------------------------------------------------------------------------
class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        ratio = logs["val_loss"] / logs["loss"]
        print(f"Epoch={epoch}, val/train={ratio:.2f}")

val_train_ratio_cb = PrintValTrainRatioCallback() 


----------------------------------------------------------------------------------
# ExponentialDecay
------------------------------------------------------------------------------
n_steps = n_epochs * math.ceil(len(X_train) / batch_size)
scheduled_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=n_steps, decay_rate=0.1)
optimizer = tf.keras.optimizers.SGD(learning_rate=scheduled_learning_rate)


# ExponentialDecay - Manually
----------------------------------------------------------------------------------
class ExponentialDecay(tf.keras.callbacks.Callback):
    def __init__(self, n_steps=40_000):
        super().__init__()
        self.n_steps = n_steps

    def on_batch_begin(self, batch, logs=None):
        # Note: the `batch` argument is reset at each epoch
        lr = K.get_value(self.model.optimizer.learning_rate)
        new_learning_rate = lr * 0.1 ** (1 / self.n_steps)
        K.set_value(self.model.optimizer.learning_rate, new_learning_rate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.learning_rate)

n_steps = n_epochs * math.ceil(len(X_train) / batch_size)
exp_decay = ExponentialDecay(n_steps)
--------------------------------------------------------------------------------------
# ExponentialLearningRate
---------------------------------------------------------------------------------------
class ExponentialLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.sum_of_epoch_losses = 0

    def on_batch_end(self, batch, logs=None):
        mean_epoch_loss = logs["loss"]  # the epoch's mean loss so far 
        new_sum_of_epoch_losses = mean_epoch_loss * (batch + 1)
        batch_loss = new_sum_of_epoch_losses - self.sum_of_epoch_losses
        self.sum_of_epoch_losses = new_sum_of_epoch_losses
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(batch_loss)
        K.set_value(self.model.optimizer.learning_rate,
                    self.model.optimizer.learning_rate * self.factor)

def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=1e-4,
                       max_rate=1):
    init_weights = model.get_weights()
    iterations = math.ceil(len(X) / batch_size) * epochs
    factor = (max_rate / min_rate) ** (1 / iterations)
    init_lr = K.get_value(model.optimizer.learning_rate)
    K.set_value(model.optimizer.learning_rate, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.learning_rate, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses, "b")
    plt.gca().set_xscale('log')
    max_loss = losses[0] + min(losses)
    plt.hlines(min(losses), min(rates), max(rates), color="k")
    plt.axis([min(rates), max(rates), 0, max_loss])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.grid()

batch_size = 128
rates, losses = find_learning_rate(model, X_train, y_train, epochs=1,
                                   batch_size=batch_size)
plot_lr_vs_loss(rates, losses)
