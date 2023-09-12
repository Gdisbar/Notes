## Preprocess large data
X = tf.range(10)  # any data tensor
X =  {"a": ([1, 2, 3], [4, 5, 6]), "b": [7, 8, 9]}
dataset = tf.data.Dataset.from_tensor_slices(X)
dataset = dataset.repeat(3).batch(7) #  batch 7 eleemnts containig 0-9 in cyclic order
# tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
# tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
# tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
# tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
# tf.Tensor([8 9], shape=(2,), dtype=int32)
dataset = dataset.map(lambda x: x * 2)  # x is a batch
dataset = dataset.filter(lambda x: tf.reduce_sum(x) > 50)
# tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int32)
# tf.Tensor([ 8 10 12 14 16 18  0], shape=(7,), dtype=int32)
# tf.Tensor([ 2  4  6  8 10 12 14], shape=(7,), dtype=int32)

## shuffling data
----------------------
dataset = tf.data.Dataset.range(10).repeat(2)
dataset = dataset.shuffle(buffer_size=4, seed=42).batch(7)
# tf.Tensor([1 4 2 3 5 0 6], shape=(7,), dtype=int64)
# tf.Tensor([9 8 2 0 3 1 4], shape=(7,), dtype=int64)
# tf.Tensor([5 7 9 6 7 8], shape=(6,), dtype=int64)
-----------------------------------------------------------------------------------------------
## split dataset into multiple part
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, 
												housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

from pathlib import Path

def save_to_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = Path() / "datasets" / "housing"
    housing_dir.mkdir(parents=True, exist_ok=True)
    filename_format = "my_{}_{:02d}.csv"

    filepaths = []
    m = len(data)
    chunks = np.array_split(np.arange(m), n_parts)
    for file_idx, row_indices in enumerate(chunks):
        part_csv = housing_dir / filename_format.format(name_prefix, file_idx)
        filepaths.append(str(part_csv))
        with open(part_csv, "w") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths
## np.c_[] = concatenation along the second axis=1
train_data = np.c_[X_train, y_train] 
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)

train_filepaths = save_to_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_csv_files(test_data, "test", header, n_parts=10)
# 20 581 --> chunk length, no of elements in each chunk
# 10 387
# 10 516
# print("".join(open(train_filepaths[0]).readlines()[:4]))
# MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,MedianHouseValue
# 3.5214,15.0,3.0499445061043287,1.106548279689234,1447.0,1.6059933407325193,37.63,-122.43,1.442
# 5.3275,5.0,6.490059642147117,0.9910536779324056,3464.0,3.4433399602385686,33.69,-117.39,1.687
# 3.1,29.0,7.5423728813559325,1.5915254237288134,1328.0,2.2508474576271187,38.44,-122.98,1.621
# train_filepaths
# ['datasets/housing/my_train_00.csv',
#  'datasets/housing/my_train_01.csv',
#  ....
#  'datasets/housing/my_train_19.csv']
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
# tf.Tensor(b'datasets/housing/my_train_05.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets/housing/my_train_16.csv', shape=(), dtype=string)
# ...
# tf.Tensor(b'datasets/housing/my_train_08.csv', shape=(), dtype=string)
n_readers = 5
dataset = filepath_dataset.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
						cycle_length=n_readers)
# tf.Tensor(b'4.5909,16.0,5.475877192982456,1.0964912280701755,1357.0,2.9758771929824563,33.63,-117.71,2.418', shape=(), dtype=string)
# tf.Tensor(b'2.4792,24.0,3.4547038327526134,1.1341463414634145,2251.0,3.921602787456446,34.18,-118.38,2.0', shape=(), dtype=string)
# tf.Tensor(b'4.2708,45.0,5.121387283236994,0.953757225433526,492.0,2.8439306358381504,37.48,-122.19,2.67', shape=(), dtype=string)
# tf.Tensor(b'2.1856,41.0,3.7189873417721517,1.0658227848101265,803.0,2.0329113924050635,32.76,-117.12,1.205', shape=(), dtype=string)
# tf.Tensor(b'4.1812,52.0,5.701388888888889,0.9965277777777778,692.0,2.4027777777777777,33.73,-118.31,3.215', shape=(), dtype=string)
---------------------------------------------------------------------------------------------------------
## Prefetching -> skip header & interleave + shuffle dataset => create data from different dataframe

def csv_reader_dataset(filepaths, n_readers=5, n_read_threads=tf.data.AUTOTUNE,
                       n_parse_threads=5, shuffle_buffer_size=10_000, seed=42,
                       batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths, seed=seed)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    return dataset.batch(batch_size).prefetch(1)


example_set = csv_reader_dataset(train_filepaths, batch_size=3)
for X_batch, y_batch in example_set.take(2):
    print("X =", X_batch)
    print("y =", y_batch)
    print()

## some methods
# sample_from_datasets()  -> Samples elements at random from the datasets in `datasets`.
# reduce()  ------>           Reduces the input dataset to a single element.
# cache()   ------>          Caches the elements in this dataset.
--------------------------------------------------------------------------------------------------
## using tf.data.Dataset + keras
train_set = csv_reader_dataset(train_filepaths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)
model.fit(train_set, validation_data=valid_set, epochs=5)
test_mse = model.evaluate(test_set)
new_set = test_set.take(3)  # pretend we have 3 new samples
y_pred = model.predict(new_set)  # or you could just pass a NumPy array

for epoch in range(n_epochs):
    for X_batch, y_batch in train_set:
        # print("\rEpoch {}/{}".format(epoch + 1, n_epochs), end="")
        with tf.GradientTape() as tape:
            y_pred = model.predict(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred)) ## defined outside
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

## using function decorator

@tf.function
def train_one_epoch(model, optimizer, loss_fn, train_set):
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.mean_squared_error
for epoch in range(n_epochs):
    #print("\rEpoch {}/{}".format(epoch + 1, n_epochs), end="")
    train_one_epoch(model, optimizer, loss_fn, train_set)

---------------------------------------------------------------------------------------------------
## TFRecord Format -> TFRecord file is just a list of binary records tpo store large files

# read multiple record
filepaths = ["my_test_{}.tfrecord".format(i) for i in range(5)]
for i, filepath in enumerate(filepaths):
    with tf.io.TFRecordWriter(filepath) as f:
        for j in range(3):
            f.write("File {} record {}".format(i, j).encode("utf-8"))

dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=tf.data.AUTOTUNE)
# compressed TFRecord
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"Compress, compress, compress!")
dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],compression_type="GZIP")
# tf.Tensor(b'Compress, compress, compress!', shape=(), dtype=string)

# storing images
feature_description = { "image": tf.io.VarLenFeature(tf.string) }
# feature_description = {
#     "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
#     "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     "emails": tf.io.VarLenFeature(tf.string),
# }
# {'emails': SparseTensor(indices=tf.Tensor(
# [[0]
#  [1]], shape=(2, 1), dtype=int64), values=tf.Tensor([b'a@b.com' b'c@d.com'], shape=(2,), dtype=string), dense_shape=tf.Tensor([2], shape=(1,), dtype=int64)), 'id': <tf.Tensor: shape=(), dtype=int64, numpy=123>, 'name': <tf.Tensor: shape=(), dtype=string, numpy=b'Alice'>}
# {'emails': SparseTensor(indices=tf.Tensor(
# [[0]
#  [1]], shape=(2, 1), dtype=int64), values=tf.Tensor([b'a@b.com' b'c@d.com'], shape=(2,), dtype=string), dense_shape=tf.Tensor([2], shape=(1,), dtype=int64)), 'id': <tf.Tensor: shape=(), dtype=int64, numpy=123>, 'name': <tf.Tensor: shape=(), dtype=string, numpy=b'Alice'>}
# {'emails': SparseTensor(indices=tf.Tensor(
# [[0]
#  [1]], shape=(2, 1), dtype=int64), values=tf.Tensor([b'a@b.com' b'c@d.com'], shape=(2,), dtype=string), dense_shape=tf.Tensor([2], shape=(1,), dtype=int64)), 'id': <tf.Tensor: shape=(), dtype=int64, numpy=123>, 'name': <tf.Tensor: shape=(), dtype=string, numpy=b'Alice'>}
# {'emails': SparseTensor(indices=tf.Tensor(
# [[0]
#  [1]], shape=(2, 1), dtype=int64), values=tf.Tensor([b'a@b.com' b'c@d.com'], shape=(2,), dtype=string), dense_shape=tf.Tensor([2], shape=(1,), dtype=int64)), 'id': <tf.Tensor: shape=(), dtype=int64, numpy=123>, 'name': <tf.Tensor: shape=(), dtype=string, numpy=b'Alice'>}
# {'emails': SparseTensor(indices=tf.Tensor(
# [[0]
#  [1]], shape=(2, 1), dtype=int64), values=tf.Tensor([b'a@b.com' b'c@d.com'], shape=(2,), dtype=string), dense_shape=tf.Tensor([2], shape=(1,), dtype=int64)), 'id': <tf.Tensor: shape=(), dtype=int64, numpy=123>, 'name': <tf.Tensor: shape=(), dtype=string, numpy=b'Alice'>}
tf.sparse.to_dense(parsed_example["emails"], default_value=b"")
# <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'a@b.com', b'c@d.com'], dtype=object)>

def parse(serialized_example):
    example_with_image = tf.io.parse_single_example(serialized_example,feature_description)
    return tf.io.decode_jpeg(example_with_image["image"].values[0])
    # or you can use tf.io.decode_image() instead
dataset = tf.data.TFRecordDataset("my_image.tfrecord").map(parse)

# serialize tensor
tensor = tf.constant([[0., 1.], [2., 3.], [4., 5.]])
serialized = tf.io.serialize_tensor(tensor)
sparse_tensor = parsed_example["emails"]
serialized_sparse = tf.io.serialize_sparse(sparse_tensor) # for spare tensor

## Protobuf