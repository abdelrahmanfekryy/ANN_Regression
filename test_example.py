import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers, optimizers, callbacks
from plot_assest import Plot_Regression



x,y = make_regression(n_samples=200, n_features=1,noise=20,random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = Sequential()
model.add(layers.InputLayer(input_shape=(1,)))
model.add(layers.Dense(1,activation='linear'))


weights_history = []
print_weights = callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_history.append(model.get_weights()))
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.01)

model.compile(optimizer=optimizers.Adam(learning_rate=0.01),loss='mse',metrics=['mae'])
model.fit(x_train, y_train, epochs=200, batch_size=1,validation_split=0.1,callbacks=[earlyStopping,print_weights], verbose=0,shuffle=False)

Plot_Regression(model,x_train,y_train,weights_history)