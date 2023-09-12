## Anomaly Detection using AutoEncoder
===================================================================================
# use case is novelty detection so use only the normal data = 1, anomaly = 0
# for training
train_index = y_train[y_train == 1].index
train_data = x_train.loc[train_index]

# min max scale the input data

class AutoEncoder(Model):
  """
  Parameters
  ----------
  output_units: int
    Number of output units
  
  code_size: int
    Number of units in bottle neck
  """

  def __init__(self, output_units, code_size=8):
    super().__init__()
    self.encoder = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(code_size, activation='relu')
    ])
    self.decoder = Sequential([
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(output_units, activation='sigmoid')
    ])
  
  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded
  
model = AutoEncoder(output_units=x_train_scaled.shape[1])
# configurations of model
model.compile(loss='msle', metrics=['mse'], optimizer='adam')

history = model.fit(
    x_train_scaled,
    x_train_scaled,
    epochs=20,
    batch_size=512,
    validation_data=(x_test_scaled, x_test_scaled)
)

# reconstructions = model.predict(x_train_scaled)
# provides losses of individual instances
# reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
# threshold for anomaly scores
threshold = np.mean(reconstruction_errors.numpy()) + 
						np.std(reconstruction_errors.numpy())
# predictions = model.predict(x_test_scaled)
# provides losses of individual instances
errors = tf.keras.losses.msle(predictions, x_test_scaled)
# 0 = anomaly, 1 = normal
anomaly_mask = pd.Series(errors) > threshold
preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)

--------------------------------------------------------------------------------
# train & test shape ['Date','Close'] : ((7059,2),(1764,2)) 

# StandardScaler() -> output is not 0-1

TIME_STEPS=30

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps+1):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['Close']], train['Close'])
X_test, y_test = create_sequences(test[['Close']], test['Close'])

# LSTM input_shape : [samples, TIME_STEPS, features]
# train shape : (7029,30,1)
# test shape : (1734,30,1)

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1])) # repeats the inputs 30 times.
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2]))) # to get the output
model.compile(optimizer='adam', loss='mae')
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
					validation_split=0.1,
                    callbacks=[keras.callbacks.EarlyStopping(
                    	monitor='val_loss', patience=3, mode='min')], 
                    shuffle=False)

# X_train_pred = model.predict(X_train, verbose=0)
# train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
threshold = np.max(train_mae_loss)
test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['Close'] = test[TIME_STEPS:]['Close']
anomalies = test_score_df.loc[test_score_df['anomaly'] == True]

-----------------------------------------------------------------------------------
# normalize the value data. We have a value for every 5 mins for 14 days.

#     24 * 60 / 5 = 288 timesteps per day
#     288 * 14 = 4032 data points in total
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
print("Number of training samples:", len(df_training_value))

# build a convolutional reconstruction autoencoder model. 
# The model will take input of shape (batch_size, sequence_length, num_features) 
# and return output of the same shape. In this case, sequence_length is 288 and 
# num_features is 1.

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

# prepare test data
df_test_value = (df_daily_jumpsup - training_mean) / training_std
x_test = create_sequences(df_test_value.values,TIME_STEPS) # defined above
anomalies = test_mae_loss > threshold
# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
# anomalous_data_indices = []
# for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
#     if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
#         anomalous_data_indices.append(data_idx)
# df_subset = df_daily_jumpsup.iloc[anomalous_data_indices]
# fig, ax = plt.subplots()
# df_daily_jumpsup.plot(legend=False, ax=ax)
# df_subset.plot(legend=False, ax=ax, color="r")
# plt.show()


================================================================================
# Handeling Data with Regular Gaps
================================================================================
# df['Date'] -> 9 to 6 everyday hourly basis except for Saturday & Sunday
# approx 10 datapoints everyday

filter1= df['day']=='Saturday'
filter2 = df['day']=='Sunday'
df['weekday'] = np.where(filter1|filter2,0,1)
#df.query("day=='Saturday' or day=='Sunday'")
df['time_bin']=pd.cut(df['Date'].dt.hour,bins=3,labels=False)
time_oh = pd.get_dummies(df['time_bin'],prefix='tbin')
df=df.join(time_oh)
# add data for paticular hrs of day
#future = future[future['ds'].dt.hour > 8][future['ds'].dt.hour<19]

==================================================================================
# same as pd.diff(window_size)
df_lag = df['Close'].rolling(window=window_size).apply(
        lambda X : x.iloc[window_size]-x.iloc[0]).dropna()

==================================================================================
# multiple timeseries Deep AR - point in time & probabilistic forecast
==================================================================================
# each row is a product & each column is a feature of that product , here m >> n
# but this is because we've transposed the data - it's better to use every model
# for training rather than using every feature for training

ts_code = df['index'].astype('category').cat.codes.values # index has product name
freq='15min'
start_train = pd.Timestamp('2020-01-01 00:15:00',freq=freq)
start_test = pd.Timestamp('2022-01-01 00:15:00',freq=freq)
prediction_length = 672 # 4*15 = 60,4*24 = 96,96*6=672

# estimator = DeepAREstimator()
train_ds = ListDataset([{
        FieldName.TARGET: target, # target : train
        FieldName.START: start_train, #for test - start_test
        FieldName.FEAT_STATIC_CAT: fsc # fsc : ts_code
    }
    for (target,fsc) in zip(train,ts_code.reshape(-1,1))],freq=freq)
# predictor = estimator.train(training_data=train)
forecast,ts = make_evaluation_prediction(dataset=test,predictor=predictor,num_samples=100)

==================================================================================
### Stationary Transformations
Seasonal Differencing
-------------------------
build model with different Differencing period 

Transformations - Tabular Data
-------------------------------
exponential Series -> log transform , quadratic Series -> sqrt transform

Lag Variables & Aggregation - AR(p) model smoothing
--------------------------------------------------------
Lag Variables using shift() + ACF & PACF

lagged_feature_cols = ['t-3', 't-2', 't-1']
# Drop first 3 rows due to NaNs
df_lagged = df.loc[3:, lagged_feature_cols + ['orders']]
# Create feature df to use for aggregation calculations
df_lagged_features = df_lagged.loc[:, lagged_feature_cols]
# Create aggregated features
df_lagged['max'] = df_lagged_features.aggregate(np.max, axis=1)
df_lagged['min'] = df_lagged_features.aggregate(np.min, axis=1)
# features_list = ['t-3', 't-2', 't-1','max','min','orders']


# features_list = ['code','date','tmp']
df['tmp_avg_7d'] = df.sort_values(by=['code','date']).set_index(
                            'date').groupby('code')['tmp'].rolling(
                            window=7, closed='both').mean()
df.reset_index(inplace=True)

#for very large data use RasgoQL


Rolling Window & Expanding Window
-----------------------------------------
window_size = 14
data['rolling_mean'] = data['Count'].rolling(window=window_size).mean()
data['expanding_mean'] = data['Count'].expanding(window_size).mean()


Custom Timestamp
-----------------------
df_lagged.index = df.date[3:]
# Create month and quarter columns
df_lagged['month'] = df_lagged.index.month 
df_lagged['quarter'] = df_lagged.index.quarter


data['dayofweek_num']=data['Datetime'].dt.dayofweek  
data['dayofweek_name']=data['Datetime'].dt.weekday_name


Encode Timestamp
------------------------
# dummy variables
----------------------------------
X_1 = pd.DataFrame(data=pd.get_dummies(X.index.month, drop_first=True, prefix="month"))
X_1.index = X.index
# feature_list = ['month_2','month_3',...,'month_12']

# cyclical encoding with sine
----------------------------------
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
X_2["month"] = X_2.index.month
X_2["month_sin"] = sin_transformer(12).fit_transform(X_2)["month"]
X_2["day_sin"] = sin_transformer(365).fit_transform(X_2)["day_of_year"]

# radial basis
-----------------------------------
rbf = RepeatingBasisFunction(n_periods=12,column="day_of_year",input_range=(1,365),
                            remainder="drop")
rbf.fit(X)
X_3 = pd.DataFrame(index=X.index,data=rbf.transform(X))