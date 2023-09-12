try:
    tf.constant(2.0) + tf.constant(40., dtype=tf.float64)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

# cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor 
# but is a double tensor [Op:AddV2]
t2 = tf.constant(40., dtype=tf.float64)
tf.constant(2.0) + tf.cast(t2, tf.float32) # 42.0

v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v.assign(2 * v)
# <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
# array([[ 2.,  4.,  6.],
#        [ 8., 10., 12.]], dtype=float32)>
v[0, 1].assign(42)
# array([[ 2., 42.,  6.],
#        [ 8., 10., 12.]], dtype=float32)>
v[:, 2].assign([0., 1.])
# array([[ 2., 42.,  0.],
#        [ 8., 10.,  1.]], dtype=float32)>
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])
# array([[100.,  42.,   0.],
#        [  8.,  10., 200.]], dtype=float32)>
sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],indices=[1, 0])
v.scatter_update(sparse_delta)
# array([[4., 5., 6.],
#        [1., 2., 3.]], dtype=float32)>
try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print(ex)
# 'ResourceVariable' object does not support item assignment
u = tf.constant([ord(c) for c in "café"])
# <tf.Tensor: shape=(4,), dtype=int32, numpy=array([ 99,  97, 102, 233], dtype=int32)>
tf.constant("café")
# <tf.Tensor: shape=(), dtype=string, numpy=b'caf\xc3\xa9'>
p = tf.constant(["Café", "Coffee", "caffè", "咖啡"])
tf.strings.length(p, unit="UTF8_CHAR")
# <tf.Tensor: shape=(4,), dtype=int32, numpy=array([4, 6, 5, 2], dtype=int32)>
------------------------------------------------------------------------------------
# Ragged tensors are the TensorFlow equivalent of nested variable-length lists.
#  They make it easy to store and process data with non-uniform shapes, 
#  including:

# Variable-length features, such as the set of actors in a movie.
# Batches of variable-length sequential inputs, such as sentences or video clips.
# Hierarchical inputs, such as text documents that are subdivided into 
# sections, paragraphs, sentences, and words.
# Individual fields in structured inputs, such as protocol buffers.

r = tf.strings.unicode_decode(p, "UTF8") #  r[1],r[2],r[3],r[4]
# <tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101],
#  [99, 97, 102, 102, 232], [21654, 21857]]>  
r2 = tf.ragged.constant([[65, 66], [], [67]])
tf.concat([r, r2], axis=0) ## add new rows
# <tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101],
#  [99, 97, 102, 102, 232], [21654, 21857], [65, 66], [], [67]]>
r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
print(tf.concat([r, r3], axis=1)) ## add new columns
# <tf.RaggedTensor [[67, 97, 102, 233, 68, 69, 70], [67, 111, 102, 102, 101, 101, 71],
#  [99, 97, 102, 102, 232], [21654, 21857, 72, 73]]>
r.to_tensor()
# <tf.Tensor: shape=(4, 6), dtype=int32, numpy=
# array([[   67,    97,   102,   233,     0,     0],
#        [   67,   111,   102,   102,   101,   101],
#        [   99,    97,   102,   102,   232,     0],
#        [21654, 21857,     0,     0,     0,     0]], dtype=int32)>
------------------------------------------------------------------------------------
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],values=[1., 2., 3.],
                    dense_shape=[3, 4])
# SparseTensor(indices=tf.Tensor(
# [[0 1]
#  [1 0]
#  [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 2. 3.], 
#  shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
tf.sparse.to_dense(s)
# <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
# array([[0., 1., 0., 0.],
#        [2., 0., 0., 0.],
#        [0., 0., 0., 3.]], dtype=float32)>
s * 42.0
# SparseTensor(indices=tf.Tensor(
# [[0 1]
#  [1 0]
#  [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([ 42.  84. 126.], 
#  shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
try:
    s + 42.0
except TypeError as ex:
    print(ex) # unsupported operand type(s) for +: 'SparseTensor' and 'float'

# extra code – shows how to multiply a sparse tensor and a dense tensor
s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
tf.sparse.sparse_dense_matmul(s, s4)
# <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
# array([[ 30.,  40.],
#        [ 20.,  40.],
#        [210., 240.]], dtype=float32)>
# extra code – when creating a sparse tensor, values must be given in "reading
#              order", or else `to_dense()` will fail.
s5 = tf.SparseTensor(indices=[[0, 2], [0, 1]],  # WRONG ORDER!
                     values=[1., 2.],
                     dense_shape=[3, 4])
try:
    tf.sparse.to_dense(s5)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
# extra code – shows how to fix the sparse tensor s5 by reordering its values
s6 = tf.sparse.reorder(s5)
tf.sparse.to_dense(s6)
# <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
# array([[0., 2., 1., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]], dtype=float32)>
------------------------------------------------------------------------------------
array = tf.TensorArray(dtype=tf.float32, size=3)
array = array.write(0, tf.constant([1., 2.]))
array = array.write(1, tf.constant([3., 10.]))
array = array.write(2, tf.constant([5., 7.]))
tensor1 = array.read(1)  # returns (and zeros out!) tf.constant([3., 10.])

array.stack()
# <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
# array([[1., 2.],
#        [0., 0.],
#        [5., 7.]], dtype=float32)>
# extra code – shows how to disable clear_after_read
array2 = tf.TensorArray(dtype=tf.float32, size=3, clear_after_read=False)
array2 = array2.write(0, tf.constant([1., 2.]))
array2 = array2.write(1, tf.constant([3., 10.]))
array2 = array2.write(2, tf.constant([5., 7.]))
tensor2 = array2.read(1)  # returns tf.constant([3., 10.])
array2.stack()
# <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
# array([[ 1.,  2.],
#        [ 3., 10.],
#        [ 5.,  7.]], dtype=float32)>
# extra code – shows how to create and use a tensor array with a dynamic size
array3 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
array3 = array3.write(0, tf.constant([1., 2.]))
array3 = array3.write(1, tf.constant([3., 10.]))
array3 = array3.write(2, tf.constant([5., 7.]))
tensor3 = array3.read(1)
array3.stack()
# <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
# array([[1., 2.],
#        [0., 0.],
#        [5., 7.]], dtype=float32)>
------------------------------------------------------------------------------------
a = tf.constant([[1, 5, 9]])
b = tf.constant([[5, 6, 9, 11]])
u = tf.sets.union(a, b)
tf.sparse.to_dense(u)
# <tf.Tensor: shape=(1, 5), dtype=int32, numpy=array([[ 1,  5,  6,  9, 11]], dtype=int32)>

a = tf.constant([[1, 5, 9], [10, 0, 0]])
b = tf.constant([[5, 6, 9, 11], [13, 0, 0, 0]])
u = tf.sets.union(a, b)
tf.sparse.to_dense(u)
# <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
# array([[ 1,  5,  6,  9, 11],
#        [-1, 10, 13, -1, -1]], dtype=int32)>
# extra code – shows how to use `tf.sets.difference()`
set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
set2 = tf.constant([[4, 5, 6], [9, 10, 0]])
tf.sparse.to_dense(tf.sets.difference(set1, set2))
# <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
# array([[2, 3, 7],
#        [7, 0, 0]], dtype=int32)>
# extra code – shows how to use `tf.sets.difference()`
tf.sparse.to_dense(tf.sets.intersection(set1, set2))
# <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
# array([[5, 0],
#        [0, 9]], dtype=int32)>
# extra code – check whether set1[0] contains 5
tf.sets.size(tf.sets.intersection(set1[:1], tf.constant([[5, 0, 0, 0]]))) > 0
# <tf.Tensor: shape=(1,), dtype=bool, numpy=array([ True])>
------------------------------------------------------------------------------------
q = tf.queue.FIFOQueue(3, [tf.int32, tf.string], shapes=[(), ()])
q.enqueue([10, b"windy"])
q.enqueue([15, b"sunny"])
q.size() # <tf.Tensor: shape=(), dtype=int32, numpy=2>
q.dequeue()
# [<tf.Tensor: shape=(), dtype=int32, numpy=10>,
#  <tf.Tensor: shape=(), dtype=string, numpy=b'windy'>]
q.enqueue_many([[13, 16], [b'cloudy', b'rainy']])
q.dequeue_many(3)
# [<tf.Tensor: shape=(3,), dtype=int32, numpy=array([15, 13, 16], dtype=int32)>,
#  <tf.Tensor: shape=(3,), dtype=string, numpy=array([b'sunny', b'cloudy', b'rainy'], dtype=object)>]
------------------------------------------------------------------------------------
## TensorFlow Functions
------------------------------------------------------------------------------------
## faster execution but come at the expense of performance and deployability.
model.compile(loss=my_mse, optimizer="nadam", metrics=[my_mae], run_eagerly=True)

## converting python function to @tf.function
--------------------------------------------------------------------------------
class MyMomentumOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.9, name="MyMomentumOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate
        self._set_hyper("decay", self._initial_decay) # 
        self._set_hyper("momentum", momentum)
    
    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "momentum")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
        momentum_var = self.get_slot(var, "momentum")
        momentum_hyper = self._get_hyper("momentum", var_dtype)
        momentum_var.assign(momentum_var * momentum_hyper - (1. - momentum_hyper)* grad)
        var.assign_add(momentum_var * lr_t)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }

---------------------------------------------------------------------------------
## Gradient Tape
---------------------------------------------------------------------------------
with tf.GradientTape() as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)  # returns tensor 36.0
try:
    dz_dw2 = tape.gradient(z, w2)  # raises a RuntimeError!
except RuntimeError as ex:
    print(ex)
# A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)

with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)  # returns tensor 36.0
dz_dw2 = tape.gradient(z, w2)  # returns tensor 10.0, works fine now!
del tape
print(dz_dw1,dz_dw2)
# (<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
#  <tf.Tensor: shape=(), dtype=float32, numpy=10.0>)

# extra code – if given a vector, tape.gradient() will compute the gradient of
#              the vector's sum.
with tf.GradientTape() as tape:
    z1 = f(w1, w2 + 2.)
    z2 = f(w1, w2 + 5.)
    z3 = f(w1, w2 + 7.)
tape.gradient([z1, z2, z3], [w1, w2])
# [<tf.Tensor: shape=(), dtype=float32, numpy=136.0>,
#  <tf.Tensor: shape=(), dtype=float32, numpy=30.0>]

# extra code – shows how to compute the jacobians and the hessians
with tf.GradientTape(persistent=True) as hessian_tape:
    with tf.GradientTape() as jacobian_tape:
        z = f(w1, w2)
    jacobians = jacobian_tape.gradient(z, [w1, w2])
hessians = [hessian_tape.gradient(jacobian, [w1, w2])
            for jacobian in jacobians]
del hessian_tape
jacobians
# [<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
#  <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]
hessians
# [[<tf.Tensor: shape=(), dtype=float32, numpy=6.0>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=2.0>],
#  [<tf.Tensor: shape=(), dtype=float32, numpy=2.0>, None]]

@tf.custom_gradient
def my_softplus(z):
    def my_softplus_gradients(grads):  # grads = backprop'ed from upper layers
        return grads * (1 - 1 / (1 + tf.exp(z)))  # stable grads of softplus

    result = tf.math.log(1 + tf.exp(-tf.abs(z))) + tf.maximum(0., z)
    return result, my_softplus_gradients
# extra code – shows that the function is now stable, as well as its gradients
x = tf.Variable([1000.])
with tf.GradientTape() as tape:
    z = my_softplus(x)

z, tape.gradient(z, [x])
# (<tf.Tensor: shape=(1,), dtype=float32, numpy=array([1000.], dtype=float32)>,
#  [<tf.Tensor: shape=(1,), dtype=float32, numpy=array([1.], dtype=float32)>])
---------------------------------------------------------------------------------
## Custom TrainingLoop
---------------------------------------------------------------------------------
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.MeanAbsoluteError()]

for epoch in range(1, n_epochs + 1):
    print(f"Epoch {epoch}/{n_epochs}")
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train_scaled, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # extra code – if your model has variable constraints
        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))

        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)

        print_status_bar(step, n_steps, mean_loss, metrics)

    for metric in [mean_loss] + metrics:
        metric.reset_states()

# Function Definitions and Graphs
# How TF Functions Trace Python Functions to Extract Their Computation Graphs
# Using Autograph To Capture Control Flow
# Handling Variables and Other Resources in TF Functions
# TFRecord Format
# Compressed TFRecord Files
# TensorFlow Protobufs