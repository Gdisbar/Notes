## Kernel Initializer
------------------------------------------------------------------------------------
def my_softplus(z):
    return tf.math.log(1.0 + tf.exp(z))

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

# def my_l1_regularizer(weights):
#     return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights):  # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


class MyL1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
#                           input_shape=input_shape),
#     tf.keras.layers.Dense(1, activation=my_softplus,
#                           kernel_initializer=my_glorot_initializer,
##                           kernel_regularizer=my_l1_regularizer,
#                           kernel_regularizer=MyL1Regularizer(0.01),
#                           kernel_constraint=my_positive_weights)
# ])
# model = tf.keras.models.load_model(
#     "my_model_with_many_custom_parts",
#     custom_objects={
##        "my_l1_regularizer": my_l1_regularizer,
#        "MyL1Regularizer": MyL1Regularizer,
#        "my_positive_weights": my_positive_weights,
#        "my_glorot_initializer": my_glorot_initializer,
#        "my_softplus": my_softplus,
#     }
# )

---------------------------------------------------------------------------------
## Custom Metrics
---------------------------------------------------------------------------------
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

## model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])

class HuberMetric(tf.keras.metrics.Mean):
    def __init__(self, threshold=1.0, name='HuberMetric', dtype=None):
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        super(HuberMetric, self).update_state(metric, sample_weight)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

# model.compile(loss=tf.keras.losses.Huber(2.0), optimizer="nadam",
# weighted_metrics=[HuberMetric(2.0)])
# sample_weight = np.random.rand(len(y_train))
# history = model.fit(X_train_scaled, y_train, epochs=2,
#                     sample_weight=sample_weight)
# (history.history["loss"][0],history.history["HuberMetric"][0] * sample_weight.mean())
# model = tf.keras.models.load_model("my_model_with_a_custom_metric_v2",
#                                    custom_objects={"HuberMetric": HuberMetric})

---------------------------------------------------------------------------------
## Custom Layers
---------------------------------------------------------------------------------
exponential_layer = tf.keras.layers.Lambda(lambda x: tf.exp(x))
# extra code – like all layers, it can be used as a function:
exponential_layer([-1., 0., 1.])
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.36787948, 1., 2.7182817 ],dtype=float32)>
 
# useful if values to predict are positive with scaling (e.g., 0.001, 10., 10000).
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(30, activation="relu", input_shape=input_shape),
#     tf.keras.layers.Dense(1),
#     exponential_layer
# ])


## Discretization Layer -> transform a numerical feature into a categorical
## feature by mapping value ranges (called bins) to categories
-------------------------------------------------------------------
age = tf.constant([[10.], [93.], [57.], [18.], [37.], [5.]])
discretize_layer = tf.keras.layers.Discretization(bin_boundaries=[18., 50.])
age_categories = discretize_layer(age)
age_categories
# <tf.Tensor: shape=(6, 1), dtype=int64, numpy=
# array([[0],
#        [2],
#        [2],
#        [1],
#        [1],
#        [0]])>

discretize_layer = tf.keras.layers.Discretization(num_bins=3)
discretize_layer.adapt(age)
age_categories = discretize_layer(age)
age_categories
# <tf.Tensor: shape=(6, 1), dtype=int64, numpy=
# array([[1],
#        [2],
#        [2],
#        [1],
#        [2],
#        [0]])>

## OneHot Encoding
-----------------------------
two_age_categories = np.array([[1, 0], [2, 2], [2, 0]])
onehot_layer(two_age_categories)
# <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
# array([[1., 1., 0.],
#        [0., 0., 1.],
#        [1., 0., 1.]], dtype=float32)>
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3, output_mode="count")
onehot_layer(two_age_categories)
# <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
# array([[1., 1., 0.],
#        [0., 0., 2.],
#        [1., 0., 1.]], dtype=float32)>
# one-hot encode each feature separately
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3,
                                                output_mode="one_hot")
tf.keras.layers.concatenate([onehot_layer(cat)
                             for cat in tf.transpose(two_age_categories)])
# <tf.Tensor: shape=(3, 6), dtype=float32, numpy=
# array([[0., 1., 0., 1., 0., 0.],
#        [0., 0., 1., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0.]], dtype=float32)>

##StringLookup Layer
-----------------------------
cities = ["Auckland", "Paris", "Paris", "San Francisco"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(cities)
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
# <tf.Tensor: shape=(4, 1), dtype=int64, numpy=
# array([[1],
#        [3],
#        [3],
#        [0]])>
str_lookup_layer = tf.keras.layers.StringLookup(output_mode="one_hot")
str_lookup_layer.adapt(cities)
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])
# <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
# array([[0., 1., 0., 0.],
#        [0., 0., 0., 1.],
#        [0., 0., 0., 1.],
#        [1., 0., 0., 0.]], dtype=float32)>
# extra code – an example using the IntegerLookup layer
ids = [123, 456, 789]
int_lookup_layer = tf.keras.layers.IntegerLookup()
int_lookup_layer.adapt(ids)
int_lookup_layer([[123], [456], [123], [111]])
# <tf.Tensor: shape=(4, 1), dtype=int64, numpy=
# array([[3],
#        [2],
#        [3],
#        [0]])>
hashing_layer = tf.keras.layers.Hashing(num_bins=10)
hashing_layer([["Paris"], ["Tokyo"], ["Auckland"], ["Montreal"]])
# <tf.Tensor: shape=(4, 1), dtype=int64, numpy=
# array([[0],
#        [1],
#        [9],
#        [1]])>

## Embedding Layer
-------------------------------------------------------------------------
ocean_prox = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(ocean_prox)
lookup_and_embed = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[], dtype=tf.string),  # WORKAROUND
    str_lookup_layer,
    tf.keras.layers.Embedding(input_dim=str_lookup_layer.vocabulary_size(),
                              output_dim=2)
])
lookup_and_embed(np.array(["<1H OCEAN", "ISLAND", "<1H OCEAN"]))
# <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
# array([[-0.01896119,  0.02223358],
#        [ 0.02401174,  0.03724445],
#        [-0.01896119,  0.02223358]], dtype=float32)>

## Text Preprocessing
---------------------------------------------------
train_data = ["To be", "!(to be)", "That's the question", "Be, be, be."]
text_vec_layer = tf.keras.layers.TextVectorization()
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])
# <tf.Tensor: shape=(2, 4), dtype=int64, numpy=
# array([[2, 1, 0, 0],
#        [6, 2, 1, 2]])>
text_vec_layer = tf.keras.layers.TextVectorization(ragged=True)
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])
#<tf.RaggedTensor [[2, 1], [6, 2, 1, 2]]>
text_vec_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])
# <tf.Tensor: shape=(2, 6), dtype=float32, numpy=
# array([[0.96725637, 0.6931472 , 0.        , 0.        , 0.        ,
#         0.        ],
#        [0.96725637, 1.3862944 , 0.        , 0.        , 0.        ,
#         1.0986123 ]], dtype=float32)>

## Pretrained Language Model Components
-------------------------------------------------------------------------------
hub_layer = tensorflow_hub.hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2")
sentence_embeddings = hub_layer(tf.constant(["To be", "Not to be"]))
sentence_embeddings.numpy().round(2)
## Image
images = sklearn.datasets.load_sample_images()["images"]
crop_image_layer = tf.keras.layers.CenterCrop(height=100, width=100)
cropped_images = crop_image_layer(images)
plt.imshow(cropped_images[0] / 255)
plt.axis("off")

## custom dense layer
--------------------------------------------------------------
class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="he_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape)  # must be at the end

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation)}

# model = tf.keras.Sequential([
#     MyDense(30, activation="relu", input_shape=input_shape),
#     MyDense(1)
# ])

class MyMultiLayer(tf.keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        print("X1.shape: ", X1.shape ," X2.shape: ", X2.shape)  # extra code
        return X1 + X2, X1 * X2, X1 / X2

    def compute_output_shape(self, batch_input_shape):
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape1, batch_input_shape1]

# inputs1 = tf.keras.layers.Input(shape=[2])
# inputs2 = tf.keras.layers.Input(shape=[2])
# MyMultiLayer()((inputs1, inputs2))
# extra code – tests MyMultiLayer with actual data 
X1, X2 = np.array([[3., 6.], [2., 7.]]), np.array([[6., 12.], [4., 3.]]) 
MyMultiLayer()((X1, X2))
# X1.shape:  (2, 2)  X2.shape:  (2, 2)
# (<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
#  array([[ 9., 18.],
#         [ 6., 10.]], dtype=float32)>,
#  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
#  array([[18., 72.],
#         [ 8., 21.]], dtype=float32)>,
#  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
#  array([[0.5      , 0.5      ],
#         [0.5      , 2.3333333]], dtype=float32)>)

## different behaviour during training & testing
class MyGaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

# model = tf.keras.Sequential([
#     MyGaussianNoise(stddev=1.0, input_shape=input_shape),
#     tf.keras.layers.Dense(30, activation="relu",
#                           kernel_initializer="he_normal"),
#     tf.keras.layers.Dense(1)
# ])

# The inputs go through a first dense layer, then through a residual block composed of
# two dense layers and an addition operation then through this same residual 
# block three more times, then through a second residual block, and the final 
# result goes through a dense output layer. 

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z

class ResidualRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)

# model = ResidualRegressor(1)
# model = tf.keras.models.load_model("my_custom_model")

## same thing using sequential API
block1 = ResidualBlock(2, 30)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu",
                          kernel_initializer="he_normal"),
    block1, block1, block1, block1,
    ResidualBlock(2, 30),
    tf.keras.layers.Dense(1)
])

class ReconstructingRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)
        self.reconstruction_mean = tf.keras.metrics.Mean(
            name="reconstruction_error")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense(n_inputs)
        self.built = True  # WORKAROUND for super().build(batch_input_shape)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        if training:
            result = self.reconstruction_mean(recon_loss)
            self.add_metric(result)
        return self.out(Z)
# model = ReconstructingRegressor(1)
