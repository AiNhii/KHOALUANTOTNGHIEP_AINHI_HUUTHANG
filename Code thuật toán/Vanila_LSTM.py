import pandas as pd
import datetime as dt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import GridSearchCV
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("/content/drive/MyDrive/dataset/NEW/data/btc.csv", parse_dates=True, index_col="formatted_date")
df.head()
# Process data
df['H-L'] = df['high'] - df['low']
# df['O-C'] = df['open'] - df['close']
ma_1 = 7
ma_2 = 14
ma_3 = 21
df[f'SMA_{ma_1}'] = df['close'].rolling(window=ma_1).mean()
df[f'SMA_{ma_2}'] = df['close'].rolling(window=ma_2).mean()
df[f'SMA_{ma_3}'] = df['close'].rolling(window=ma_3).mean()

df[f'SD_{ma_1}'] = df['close'].rolling(window=ma_1).std()
df[f'SD_{ma_3}'] = df['close'].rolling(window=ma_3).std()
df.dropna(inplace=True)

df.to_csv("bitcoin_processed_7_3.csv")
df
pre_day = 7
scala_x = MinMaxScaler(feature_range=(0,1))
scala_y = MinMaxScaler(feature_range=(0,1))
cols_x = ['high', 'low', 'open', 'H-L', f'SMA_{ma_1}', f'SMA_{ma_2}', f'SMA_{ma_3}', f'SD_{ma_1}', f'SD_{ma_3}']
cols_y = ['close']
scaled_data_x = scala_x.fit_transform(df[cols_x].values)
scaled_data_y = scala_y.fit_transform(df[cols_y].values)

x_total = []
y_total = []

for i in range(pre_day, len(df)):
    x_total.append(scaled_data_x[i-pre_day:i])
    y_total.append(scaled_data_y[i])

test_size = (int)(len(scaled_data_y) * 0.2)
print(test_size)

x_train = np.array(x_total[:len(x_total)-test_size])
x_test = np.array(x_total[len(x_total)-test_size:])
y_train = np.array(y_total[:len(y_total)-test_size])
y_test = np.array(y_total[len(y_total)-test_size:])



print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

def create_model(drop,input_shape):
  model = Sequential()
  model.add(LSTM(50, activation='relu', input_shape=input_shape))
  # Dropout(drop)
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  return model
# model.fit(x_train, y_train, epochs=500, batch_size=32)

param_grid = {
    'drop': [0.1,0.15,0.2, 0.3, 0.4],
    'input_shape': [(3,9), (5,9), (7,9), (10,9), (20,9)]
}
model = KerasRegressor(build_fn=create_model, verbose=0)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X=x_train, y=y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best parameter: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
final_model = create_model(drop=best_params['drop'], input_shape=best_params['input_shape'])
final_model.fit(x_train, y_train, epochs=500, batch_size=32)
# Testing
predict_price = final_model.predict(x_test)
predict_price = scala_y.inverse_transform(predict_price)
# Ploting the stat
real_price = df[len(df)-test_size:]['close'].values.reshape(-1,1)
real_price = np.array(real_price)
print(real_price.shape)
real_price = real_price.reshape(real_price.shape[0], 1)

plt.figure(figsize=(16,9))
plt.grid(True)
plt.plot(real_price, color="red", label=f"Real BTC Prices")
plt.plot(predict_price, color="blue", label=f"Predicted BTC Prices")
plt.title(f"BTC Prices")
plt.xlabel("Time (day)")
plt.ylabel("Stock Prices")
plt.ylim(bottom=0)
plt.legend()
plt.show()
# Make Prediction
x_predict = df[len(df)-pre_day:][cols_x].values.reshape(-1, len(cols_x))
x_predict = scala_x.transform(x_predict)
x_predict = np.array(x_predict)
x_predict = x_predict.reshape(1, x_predict.shape[0], len(cols_x))

prediction = final_model.predict(x_predict)
prediction = scala_y.inverse_transform(prediction)
print(prediction)
mae = mean_absolute_error(real_price, predict_price)
mape = mean_absolute_percentage_error(real_price, predict_price)
mse = mean_squared_error(real_price, predict_price)
rmse = np.sqrt(mse)
r2 = r2_score(real_price, predict_price)

print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape * 100:.4f}%")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R-Squared: {r2*100:.4f}%")