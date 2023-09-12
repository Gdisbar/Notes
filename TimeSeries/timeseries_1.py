----------------------------------------------------------------------------------
## adding holiday feature in Prophet
----------------------------------------------------------------------------------
holiday_sale = pd.DataFrame({
	'holiday':'dec_jan_sale',
	'ds' : pd.to_datetime(['2017-12-31','2016-12-31',...,'2012-12-31']),
	'lower_window':0,
	'upper_window':1
})
# non-stationary + no-trend = additive ,non-stationary+trend = multiplicative
model = Prophet(interval_width=0.9,seasonality_mode='multiplicative',
						holidays=holiday_sale)

----------------------------------------------------------------------------------
## Feature Engineering
-----------------------------------------------------------------------------------
df['Date_month']=df['Date'].dt.months # extract monthh from data
df.query('temp<=80')[['y','temp']].corr()
df['summer_temp']=df['temp'].apply(lambda x : 1 if x > 0 else 0) #axis=1 not needed
df['month_bins'] = pd.cut(df['Date_month'],bins=3,labels=False)
# orginal model has yearly seasonality + y = demand , here temp is additional feature
model.add_regressor('summer_temp',standardize=False)
model.add_regressor('month_bins',standardize=False,mode='multiplicative')
# don't forrget to populate future['summer_temp'] & future['month_bins']

-----------------------------------------------------------------------------------
## Data preparation for Deep Learning Models
-----------------------------------------------------------------------------------
# remove outliers by shifting mean & std -> MinMaxScaler() not StandardScaler()

from keras.preprocessing.sequence import TimeseriesGenerator

features = df[['Appliances','T_out','RH_2']].to_numpy().tolist()
target = df['Appliances'].tolist()
# predict next i+offset value : input = 6,offset = 1, label = 1 , total  7
# predict future i+offset value : input = 24,offset=24,label=1, total = 48

ts_generator = TimeseriesGenerator(features,target,length=6,
								sampling_rate=1,batch_size=1) 
# t0-t6,t1-t7,t2-t8,... are the batches
# window = 3,stride = 1 -> 1,2,3 | 2,3,4 | 4,5,6
# window = 3,stride = 2 -> 1,2,3 | 3,4,5 | 7,8,9
# tumbling window = 3 -> 1,2,3 | 4,5,6 | 7,8,9

# printing multiple outputs
multi_target = pd.concat([df['Appliances'],df['Appliances'].shift(-1),
				df['Appliances'].shift(-2)],axis=1).dropna().to_numpy().tolist()

# row : current - next - next of next
ts_generator = TimeseriesGenerator(features[:-2],multi_target,length=6,
								sampling_rate=1,batch_size=1,stride=6)


----------------------------------------------------------------------------------
## Apply LSTM
----------------------------------------------------------------------------------
# define window_length,num_features,batch_size
# train_test_split can be used with shuffle=False & random_state=constant after
# that use TimeseriesGenerator() , batch_size = 16*n
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(128,input_shape=(window_length,num_features),
				return_sequence=True)) # need hidden state for all input
model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
model.add(tf.keras.layers.LSTM(64,return_sequence=False)) # hidden state for 1 input
model.add(tf.keras.layers.Dropout(0.2))
mode.add(tf.keras.layers.Dense(1))
model.summary()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
						patience=2,mode='min') # stop if no change in 2 iteration
model.compile(loss=tf.keras.MeanSquaredError(),optimizer=tf.optimizers.Adam(),
				metrics=[tf.metrics.MeanAbsoluteError()])
history = model.fit_generator(train_generator,epochs=50,
			validation_data=validation_generator,shuffle=False,
			callbacks=[early_stopping])
model.evaluate_generator(test_generator,verbose=0)
y_test_pred = model.predict_generator(test_generator)
df_pred = pd.concat([pd.DataFrame(y_test_pred),
		pd.DataFrame(x_test[:,1:][window_length])],axis=1)
rev_trans = scaler.inverse_transform(df_pred) # as we did MinMaxScaler()
df_final = df[y_test_pred.shape[0]*-1:] # take orginal value of y_test_pred from last
df_final['Appliances_pred'] = rev_trans[:,0] # insert predicted value

------------------------------------------------------------------------------------
### DeepAR & Gluon TS
------------------------------------------------------------------------------------
## plotting -> nrows = m/k,ncols=k,figsize=(a,m*k) , axes=[i//2,i%2]

train_time = '2017-12-01 00:00:00'
pred_length = 144 # 10 min * 6 * 24
# can use other estimators too
estimator = DeepAREstimator(freq='10min',num_hidden_dimensions=[10],
						    prediction_length=pred_length,
						    context_length=5*pred_length,
						    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, 
	    					num_batches_per_epoch=5*pred_length),
						)
# estimator = SimpleFeedForwardEstimator(
#     num_hidden_dimensions=[10],
#     prediction_length=dataset.metadata.prediction_length,
#     context_length=100,
#     trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, 
#     	num_batches_per_epoch=100),
# )


training_data = ListDataset([{'start':df.index[0],
					'target':df['Appliances'][:train_time]}],freq='10min')

predictor = estimator.train(training_data=training_data)

test_data = ListDataset([
	{'start':df.index[0],'target':df['Appliances']['2017-12-02 00:00:00']},
	{'start':df.index[0],'target':df['Appliances']['2017-12-06 00:00:00']}
	],freq='10min')

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,  
    predictor=predictor, 
    num_samples=100,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)

# forecast_entry = forecasts[0]
# ts_entry = tss[0]
# plot_length = pred_length
# pred_intervals = (80,95)
# forecast_entry.plot(prediction_intervals=pred_intervals)
# ts_entry.[-plot_length:].plot()

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(tss, forecasts)

----------------------------------------------------------------------------------
## multiple timeseries - Prophet
----------------------------------------------------------------------------------
df = df.set_index('Date').groupby('station').resample('D').mean() # data is hourly
df.set_index('Date').query("station=='Delhi'")[['Ozone','TEMP']].plot()
# split to train_test by station group
stations = df.groupby('station')
for station in stations.group:
	group = stations.get_group(station)
	train = group[(group['ds']>='2013-03-01')&(group['ds']<='2016-02-28')]
	test = group[group['ds']>'2016-02-28']
# train model
target = pd.DataFrame()
for station in stations.group:
	group = stations.get_group(station)
	model = Prophet(interval_width=0.9)
	model.fit(group)
	future = model.make_future_dataframe(periods=366)
	forecast = model.predict(future)
	model.plot(forecast)
	forecast = forecast.rename(columns={'yhat':'yhat_'+station})
	target = pd.merge(target.forecast.set_index('ds'),how='outer',
						left_index=True,right_index=True)

# plot by grop predictions
target = target[['yhat_'+station for station in stations.groups.keys()]]
pd.concat([df.set_index('ds').query("station='Delhi")['y'],
			target['yhat_Delhi']],axis=1).plot()


----------------------------------------------------------------------------------
## Anomaly Detection
----------------------------------------------------------------------------------
# global & contextual

result = pd.concat([train.set_index('ds')['y'], 
		forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]], axis=1)
result['error'] = result['y'] - result['yhat']
result['uncertainty'] = result['yhat_upper'] - result['yhat_lower']
result['anomaly'] = result.apply(
	lambda x: 'Yes' if(np.abs(x['error']) > 1.5*x['uncertainty']) else 'No', 
	axis = 1)
# change data to weekly with particular order (starting from monday)
df['hour']=df.timestamp.dt.hour
df['weekday']=pd.Categorical(df.timestamp.dt.strftime('%A'),
	catergories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
	order=True)
# plot the effect
df[['value','weekday']].groupby('weekday').mean().plot()
df[['value','hour']].groupby('hour').mean().plot()

## Isolation Forest
model = IsolationForest(contamination=0.004)
model.fit(df[['value']])
df['outliers']=pd.Series(model.predict(df[['value']])).apply(
						lambda x : 'yes' if (x==-1) else 'no')
df.query("outliers='yes'")
df['score'] = model.decision_function(df[['value']])
# plot histogram & find threshold for outliers

-----------------------------------------------------------------------------------
### multiple timeseries with pyspark
-----------------------------------------------------------------------------------
# install pyspark & pyarrow

spark = SparkSession.builder.master('local').getOrCreate()
df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
# df.query('store_id=22')[['sales']].plot()
sdf = spark.createDataFrame(df)
# sdf.show(5)
# sdf.printSchema()
# sdf.count()

sdf.select(['store_id']).groupby('store_id').agg({'store_id':'count'}).show()
sdf.createOrReplaceTempView('sales')
#sql = 'select store_id,count(*) from sales group by store_id order by store_id'
sql='select store_id, date as ds, sum(sales) as y from sales group by store_id, ds order by store_id, ds'
spark.sql(sql).show()
sdf.explain()
sdf.rdd.getNumPartitions()
store_part = (spark.sql(sql).repartition(
	spark.sparkContext.defaultParallelism,['store_id'])).cache()
# store_part.explain()
from pyspark.sql.types import *
result_schema = StructType([
                  StructField('ds', TimestampType()),
                  StructField('store_id', IntegerType()),
                  StructField('y', DoubleType()),
                  StructField('yhat', DoubleType()),
                  StructField('yhat_upper', DoubleType()),
                  StructField('yhat_lower', DoubleType())
])

from pyspark.sql.functions import pandas_udf, PandasUDFType
@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def forecast_sales(store_pd):
  model = Prophet(interval_width=0.95, seasonality_mode= 'multiplicative', weekly_seasonality=True, yearly_seasonality=True)
  model.fit(store_pd)
  future_pd = model.make_future_dataframe(periods=5, freq='w')
  forecast_pd = model.predict(future_pd)
  f_pd = forecast_pd[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].set_index('ds')
  st_pd = store_pd[['ds', 'store_id', 'y']].set_index('ds')
  result_pd = f_pd.join(st_pd, how='left')
  # replace ds(0-th level) with unique store_id
  result_pd.reset_index(level=0, inplace=True)
  result_pd['store_id'] = store_pd['store_id'].iloc[0]
  return result_pd[['ds', 'store_id', 'y', 'yhat', 'yhat_upper', 'yhat_lower']]

from pyspark.sql.functions import current_date
results = (store_part.groupby('store_id').apply(forecast_sales).withColumn('training_date', current_date()))
results.cache()
results.show()

results.coalesce(1) # opposite of repartition
print(results.count())
results.createOrReplaceTempView('forecasted')
sql = 'select store_id, count(*) from forecasted group by store_id'
spark.sql(sql).show()

final_df = results.toPandas()
final_df.query('store_id == 41')[['y', 'yhat']].plot()
plt.show()

-----------------------------------------------------------------------------------
### multivariate VAR
-----------------------------------------------------------------------------------
## H0 : series has unit root,hence non-stationary
## H1 : series stationary

for i in range(len(df.columns)):
	result = adfuller(df[df.columns[i]])
	result[1] > 0.05: # not stationary

## H0 : x[t] doesn't granger cause y[t]
## H1 : x[t] granger cause y[t]
max_lags = 8
for i in range(len(df.columns)-1):
	result = grangercausalitytest(df[['Appliances',df.columns[i+1]]],max_lags)
	p_values = [round(result[i+1][0]['ssr_ftest'][1],4)for i in range(max_lags)]
	print(df.columns[i+1],p_values)
	# p_value > 0.05 reject H0


model = VAR(df_train,freq='1H')
for i in range(48):
	results = model.fit(i+1) # order=i+1,AIC,BIC score
model.select_order(48).summary() # take lowest BIC score lag, say = 7
results = model.fit(7)
model.summary()
lag = results.k_ar
results.forecast(df_train.values[-lag:],steps=5)
# check df_test[0:5]
## Building own Equations
df_coeff = pd.DataFrame([results.params['Appliances'],
				results.p_values['Appliances']]).T
df_coeff.columns = ['coeff','pval']
df_coeff['valid'] = np.where(df_coeff['pval']<0.05,1,0)

coeff_arr = df_coeff['coeff'][1:].values # 3 feature * 7 lag = 21 coeff_arr size
in_arr = df_train[-lag:][::-1].stack().to_frame().T.values # shape (1,21)
result = np.dot(in_arr,coeff_arr)+df_coeff['coeff'][:1].values 
# next forecast for 8-th position -> 1st value in results.forecast(df_train.values[-lag:],steps=5)
