gn_vec_path = "GoogleNews-vectors-negative300.bin" # pre-trained word embedding model
if not os.path.exists("GoogleNews-vectors-negative300.bin"):
    if not os.path.exists("../Ch2/GoogleNews-vectors-negative300.bin"):
        #Downloading the reqired model
        if not os.path.exists("../Ch2/GoogleNews-vectors-negative300.bin.gz"):
            if not os.path.exists("GoogleNews-vectors-negative300.bin.gz"):
                wget.download("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz")
            gn_vec_zip_path = "GoogleNews-vectors-negative300.bin.gz"
        else:
            gn_vec_zip_path = "../Ch2/GoogleNews-vectors-negative300.bin.gz"
        #Extracting the required model
        with gzip.open(gn_vec_zip_path, 'rb') as f_in:
            with open(gn_vec_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        gn_vec_path = "../Ch2/" + gn_vec_path

print(f"Model at {gn_vec_path}")

warnings.filterwarnings("ignore") 

import psutil #This module helps in retrieving information on running processes and system resource utilization
process = psutil.Process(os.getpid())
from psutil import virtual_memory
mem = virtual_memory()



from gensim.models import Word2Vec, KeyedVectors
pretrainedpath = gn_vec_path

#Load W2V model. This will take some time, but it is a one time effort! 
pre = process.memory_info().rss
start_time = time.time() #Start the timer
ttl = mem.total #Toal memory available

w2v_model = KeyedVectors.load_word2vec_format(pretrainedpath, binary=True)


nlp = spacy.load('en_core_web_md')
# process a sentence using the model
mydoc = nlp("Canada is a large country")
#Get a vector for individual words
#print(doc[0].vector) #vector for 'Canada', the first word in the text 
print(mydoc.vector) #Averaged vector for the entire sentence



----------------------------
os.makedirs('data/en', exist_ok= True)
file_name = "data/en/enwiki-latest-pages-articles-multistream14.xml-p13159683p14324602.bz2"
file_id = "11804g0GcWnBIVDahjo5fQyc05nQLXGwF"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if not os.path.exists(file_name):
    download_file_from_google_drive(file_id, file_name)
else:
    print("file already exists, skipping download")

print(f"File at: {file_name}")
=====================================================================================
path='primary path'
files = os.listdir(path)
categories = []
for file in files:
    category = file.split(".")[0]
    if category=='dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({"filename":files,"category":categories})
df.category = df.category.replace({1:"dog",0:"cat"})
from keras.preprocessing import image
img = image.load_image(path,target_size=IMAGE_SHAPE)
#IMAGE_SHAPE = (IMAGE_WIDTH,IMAGE_HEIGHT)
#IMAGE_TENSOR = ()
img = image.img_to_array(img).reshape(1,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNEL)

datagen = image.ImageDataGenerator(
                rescale=1/255.0,       # Normalize

                horizontal_flip=False,
                rotation_range=20          #Augmentation
          ) 
data_generator = datagen.flow_from_dataframe(
                    data, #flow_from_directory -> don't need it
                    path,
                    x_col="filename",
                    y_col="category",
                    target_size=IMAGE_SIZE,
                    class_mode="categorical",  #flow_from_directory-> binary
                    batch_size=BATCH_SIZE
            )
#data_generator = datagen.flow_from_directory(...)
for x,y in data_generator:
    img=x[0]

datagen.fit(X_train)
# flow() -> Takes data & label arrays, generates batches of augmented data.
model.fit(datagen.flow(X_train,y_train,batch_size=BATCH_SIZE,subset="training"),
        validation_data=datagen.flow(X_train,y_train,batch_size=BATCH_SIZE/4,subset="validation"),
        steps_per_epoch=len(X_train)/BATCH_SIZE,epochs=EPOCHS
    )
# for e in range(EPOCHS):
#     print('Epoch', e)
#     batches = 0
#     for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=BATCH_SIZE):
#         model.fit(X_batch, y_batch)
#         batches += 1
#         if batches >= len(X_train) / BATCH_SIZE:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break


label_map = dict(k,v) for k,v in data_generator.class_indices.items()
df["category"].replace(label_map)
=========================================================================

data_dir = tf.keras.utils.get_file(origin=dataset_url,fname='flower_photos',untar=True)
data_dir = pathlib.Path(data_dir)

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
  )
class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Add the image to a batch.
image = tf.cast(tf.expand_dims(image, 0), tf.float32)

-------------------------------------------------------------------------------------
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True, # samplewise_center
                     featurewise_std_normalization=True, # samplewise_std_normalization
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)
image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)
mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)
# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)
model.fit(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)


datage.fit(X,argument=False,round=1,seed=123)
datagen.apply_transform(X,transform_parameters)
#Generates random parameters for a transformation.
datagen.get_random_transform(img_shape=IMAGE_SHAPE,seed=123) 

model.add(Conv2D(filters=32,kernel_size=(3,3),stride=(2,2),
    padding="same/valid",activation="elu",input_shape=IMAGE_SHAPE))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.compile(loss="sparse_categorical_crossentropy",optimizer=rmsprop,
                metrics=["accuracy"])
model.fit(data_generator,ecpochs=EPOCHS,validation_data=data_generator,
    validation_steps=validate.shape[0]//BATCH_SIZE,
    steps_per_epoch=train.shape[0]//BATCH_SIZE,
    callbacks=[EarlyStopping(patience=2),ReduceLROnPleatu()])
model.predict(data_generator,steps=np.ceil(test.shape[0]/BATCH_SIZE))

=====================================================================================
input_shape = (image_width,image_height,image_channel)
pre_trained_model = applications.VGG16(input_shape=input_shape, 
                include_top=False, weights="imagenet")

for layer in pre_trained_model.layers[:15]: # Freeze all layers upto last 15
    layer.trainable = False

for layer in pre_trained_model.layers[15:]: # Un-Freeze last 15
    layer.trainable = True

                                                                 
#  block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                     
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output


# Flatten the output layer to 1 dimension
x = GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

# loss, accuracy = model.evaluate(validation_generator,
#            total_validate//batch_size, workers=12) 
loss, accuracy = model.evaluate(validate_generator, workers=12)
y_val = validate['category']
y_pred =  model.predict(validate_generator)
=====================================================================================
# image_list, label_list = [], []
# try:
#     print("[INFO] Loading images ...")
#     root_dir = listdir(directory_root)
#     for directory in root_dir :
#         # remove .DS_Store from list
#         if directory == ".DS_Store" :
#             root_dir.remove(directory)

#     for plant_folder in root_dir :
#         plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
#         for disease_folder in plant_disease_folder_list :
#             # remove .DS_Store from list
#             if disease_folder == ".DS_Store" :
#                 plant_disease_folder_list.remove(disease_folder)

#         for plant_disease_folder in plant_disease_folder_list:
#             print(f"[INFO] Processing {plant_disease_folder} ...")
#             plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
#             for single_plant_disease_image in plant_disease_image_list :
#                 if single_plant_disease_image == ".DS_Store" :
#                     plant_disease_image_list.remove(single_plant_disease_image)

#             for image in plant_disease_image_list[:200]:
#                 image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
#                 if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
#                     image_list.append(convert_image_to_array(image_directory))
#                     label_list.append(plant_disease_folder)
#     print("[INFO] Image loading completed")  
# except Exception as e:
#     print(f"Error : {e}")
---------------------------------------------------------------------------------
labels = []
for path in glob.glob(f"{root_dir}/*"):
  label = path.split('/')
  labels.append(label[-1])

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

images=[]
image_labels = []
for label in labels:
  image_label = root_dir+"/"+label
  print(f"checking {image_label}")
  i=0
  for path in tqdm(glob.glob(f"{root_dir}/{label}/*")):
    
    if path.endswith(".jpg") or path.endswith(".JPG"):
      i+=1
      if i>150: break
      images.append(convert_image_to_array(path))
      image_labels.append(image_label)

image_list = np.array(images, dtype=np.float16) / 225.0

# checking /content/PlantVillage/Tomato_healthy
#   9%|▉         | 150/1591 [00:00<00:02, 528.19it/s]
# checking /content/PlantVillage/Potato___Early_blight
#  15%|█▌        | 150/1000 [00:00<00:01, 549.30it/s]
# checking /content/PlantVillage/Potato___Late_blight
print(image_labels[0].shape)
print(image_list[0].shape)
print(image_labels.shape)
print(image_list.shape)
# (15,)
# (256, 256, 3)
# (2250, 15)
# (2250, 256, 256, 3)


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = '../input/plantvillage/'
width=256
height=256
depth=3


label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
# pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype=np.float16) / 225.0

inputShape = (height, width, depth)
chanDim = -1

if backend.K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
# model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

==================================================================================
class DataGenerator(keras.utils.Sequence):
    def __init__(self, images, labels=None, mode='fit', batch_size=batch_size, dim=(height, width), channels=channels, n_classes=n_classes, shuffle=True, augment=False):
        
        #initializing the configuration of the generator
        self.images = images
        self.labels = labels
        self.mode = mode
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
   
    #method to be called after every epoch
    def on_epoch_end(self):
        self.indexes = np.arange(self.images.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    #return numbers of steps in an epoch using samples & batch size
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
    
    #this method is called with the batch number as an argument to #obtain a given batch of data
    def __getitem__(self, index):
        #generate one batch of data
        #generate indexes of batch
        batch_indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]
        
        #generate mini-batch of X
        X = np.empty((self.batch_size, *self.dim, self.channels))        for i, ID in enumerate(batch_indexes):
            #generate pre-processed image
            img = self.images[ID]
            #image rescaling
            img = img.astype(np.float32)/255.
            #resizing as per new dimensions
            img = resize_img(img, self.dim)
            X[i] = img
            
        #generate mini-batch of y
        if self.mode == 'fit':
            y = self.labels[batch_indexes]
            
            #augmentation on the training dataset
            if self.augment == True:
                X = self.__augment_batch(X)
            return X, y
        
        elif self.mode == 'predict':
            return X
        
        else:
            raise AttributeError("The mode should be set to either 'fit' or 'predict'.")
            
    #augmentation for one image
    def __random_transform(self, img):
        composition = albu.Compose([albu.HorizontalFlip(p=0.5),
                                   albu.VerticalFlip(p=0.5),
                                   albu.GridDistortion(p=0.2),
                                   albu.ElasticTransform(p=0.2)])
        return composition(image=img)['image']
    
    #augmentation for batch of images
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i] = self.__random_transform(img_batch[i])
        return img_batch

train_data_generator = DataGenerator(X_train_data, y_train_data, augment=True) 
valid_data_generator = DataGenerator(X_val_data, y_val_data, augment=False)




===============================================================================
efficient_net = EfficientNetB0(
    weights='imagenet',
    input_shape=(224,224,3),
    include_top=False,
    pooling='max'
)
efficient_net.trainable = False
model = Sequential()
model.add(efficient_net)
#model.add(GlobalMaxPooling2D(name="gap"))
model.add(Dense(units = 120, activation='relu',name="dense_1_out"))

model.add(Dropout(rate=0.2, name="dropout_out"))
model.add(Dense(units = 120, activation = 'relu',name="dense_2_out"))
model.add(Dense(units = 15, activation='softmax',name="fc_out"))
model.summary()

images=[]
image_labels = []

for label in labels:
  image_label = root_dir+"/"+label
  print(f"checking {image_label}")
  i=0
  for path in tqdm(glob.glob(f"{root_dir}/{label}/*")):
    
    if path.endswith(".jpg") or path.endswith(".JPG"):
      i+=1
      if i>150: break
      image = cv2.imread(path)
      if image is not None:
        image = cv2.resize(image,(224,224))
        images.append(path)
        image_labels.append(label)

from sklearn import preprocessing

df=pd.DataFrame(images,columns=['image'])
df['class']=image_labels
df = df.sample(frac = 1)



label_encoder = preprocessing.LabelEncoder()
df['class_id']= label_encoder.fit_transform(df['class'])
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
df['class_id']=df['class_id'].astype('str')

print(category_mapping)
df.head(5)



#category_ids = df.category_id.unique()

TRAIN_IMAGES_PATH='/content/train'
VAL_IMAGES_PATH='/content/val'
os.makedirs(TRAIN_IMAGES_PATH,exist_ok=True)
os.makedirs(VAL_IMAGES_PATH,exist_ok=True)


for class_id in [x for x in range(len(labels))]:
  os.makedirs(os.path.join(TRAIN_IMAGES_PATH, str(class_id)),exist_ok=True)
  os.makedirs(os.path.join(VAL_IMAGES_PATH, str(class_id)),exist_ok=True)

from sklearn.model_selection import train_test_split

def preprocess_data(df,image_path):
  for index,row in tqdm(df.iterrows(),total=len(df)):
    idx = row["class_id"]
    shutil.copy(row["image"],os.path.join(image_path, str(idx)))

df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
preprocess_data(df_train, TRAIN_IMAGES_PATH)
preprocess_data(df_valid, VAL_IMAGES_PATH)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)# Note that the validation data should not be augmented!#and a very important step is to normalise the images through  rescaling
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    TRAIN_IMAGES_PATH,
    # All images will be resized to target height and width.
    target_size=(224,224),
    batch_size=16,
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode="categorical",
)
validation_generator = test_datagen.flow_from_directory(
    VAL_IMAGES_PATH,
    target_size=(224,224),
    batch_size=16,
    class_mode="categorical",
)


NUMBER_OF_TRAINING_IMAGES = 1800
NUMBER_OF_VALIDATION_IMAGES = 450
batch_size = 16
epochs = 10

model.compile(
    loss="categorical_crossentropy",
    optimizer=RMSprop(learning_rate=2e-5),
    metrics=["accuracy"],
)

history = model.fit(
    train_generator,
    steps_per_epoch=NUMBER_OF_TRAINING_IMAGES // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=NUMBER_OF_VALIDATION_IMAGES // batch_size,
    verbose=1,
    use_multiprocessing=True,
    workers=4,
)
