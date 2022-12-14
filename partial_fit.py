# https://www.kaggle.com/competitions/grupo-bimbo-inventory-demand/overview
# Feature Prediction competition

Feature hashing trick with partial fitting (low memore usage)
===============================================================
"""
train.csv 
Agencia_ID: # of unique = 552 
Canal_ID: # of unique = 9
Ruta_SAK: # of unique = 3603 
Cliente_ID: # of unique = 880604
Producto_ID: # of unique = 1799

"""

#   rmsle - error function used in LB
def rmsle_func(actual, predicted):
    return np.sqrt(msle(actual, predicted))
def msle(actual, predicted):
    return np.mean(sle(actual, predicted))
def sle(actual, predicted):
    return (np.power(np.log(np.array(actual)+1) - np.log(np.array(predicted)+1), 2))

# to create post only non negative values
    def nonnegative(x):
        if x > 0:
            return x
        else: 
            return 0

#  to decrease memory usage:          
dtypes = {'Semana' : 'int32',
          'Agencia_ID' :'int32',
          'Canal_ID' : 'int32',
          'Ruta_SAK' : 'int32',
          'Cliente-ID' : 'int32',
          'Producto_ID':'int32',
          'Venta_hoy':'float32',
          'Venta_uni_hoy': 'int32',
          'Dev_uni_proxima':'int32',
          'Dev_proxima':'float32',
          'Demanda_uni_equil':'int32'}

model = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, 
                     fit_intercept=True, n_iter=10, shuffle=True, verbose=0, 
                     epsilon=0.1, learning_rate='invscaling', 
                     eta0=0.01, power_t=0.25, warm_start=True, average=False)

from sklearn.feature_extraction import FeatureHasher
h = FeatureHasher(n_features=8000, input_type = 'string') 
#8000 - the number of total unique values over all data


df_train = pd.read_csv('train.csv', dtype  = dtypes, 
            usecols=["Semana", "Agencia_ID", "Canal_ID", 'Ruta_SAK',
                     'Producto_ID','Demanda_uni_equil'], chunksize=1500)

# use astype('str')  in python 3
i = 1
num = 100# to limit train size
#pd.concat([train, pd.get_dummies(train['Semana'],sparse=True)], 
#            axis=1, join_axes=[train.index])
for chunk in df_train:
    if  i < num :
        X_chunk = h.fit_transform(chunk[["Semana", "Agencia_ID", "Canal_ID", 
                    'Ruta_SAK', 'Producto_ID']].astype('string').as_matrix())
        y_chunk = np.ravel(chunk[['Demanda_uni_equil']].as_matrix())

        model.partial_fit(X_chunk, y_chunk)
        i = i + 1
    elif i == num:
        X_chunk = h.fit_transform(chunk[["Semana", "Agencia_ID", "Canal_ID", 
            'Ruta_SAK','Producto_ID']].astype('string').values)
        y_chunk = np.ravel(chunk[['Demanda_uni_equil']].values)

        print ('rmsle: ', rmsle_func(y_chunk, model.predict(X_chunk)))
        print ('RMSE ', math.sqrt(sklearn.metrics.mean_squared_error(y_chunk, 
                        model.predict(X_chunk))))
        i = i + 1
    else:
        break
print('Finished the fitting')

#predict

    X_test = pd.read_csv('../input/test.csv',dtype  = dtypes,
                 usecols=['id', "Semana", "Agencia_ID", "Canal_ID", 
                           'Ruta_SAK','Producto_ID'])
ids = X_test['id']
X_test.drop(['id'], axis =1, inplace = True)

y_predicted = model.predict(h.fit_transform(X_test.astype('string').values))



# submission   
# submission = pd.DataFrame({"id":ids, "Demanda_uni_equil": y_predicted})

# y_predicted = map(nonnegative, y_predicted)

# submission = pd.DataFrame({"id":ids, "Demanda_uni_equil": y_predicted})
# cols = ['id',"Demanda_uni_equil"]
# submission = submission[cols]
# submission.to_csv("submission.csv", index=False)


Partial Dependence Plot - Plotting single dependent feature again independent feature
======================================================================================
#https://www.kaggle.com/code/dansbecker/partial-dependence-plots
"""
The relationship (according to our model) between Price and a couple variables 
from the Melbourne Housing dataset. We''ll walk through how these plots are 
created and interpreted.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer

cols_to_use = ['Distance', 'Landsize', 'BuildingArea']

def get_some_data():
    data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')
    y = data.Price
    X = data[cols_to_use]
    my_imputer = Imputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y
    

X, y = get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
my_plots = plot_partial_dependence(my_model, 
                                   features=[0,2], 
                                   X=X, 
                                   feature_names=cols_to_use, 
                                   grid_resolution=10)

# Plot 0 - 'Distance' vs 'Price' , Plot 1 - 'BuildingArea' vs 'Price'

