import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Housing/housePrices.csv")
data['Living_Space_sq_ft'] = 2 *(data['Living_Space_sq_ft'] - min(data['Living_Space_sq_ft'])) / (max(data['Living_Space_sq_ft']) - min(data['Living_Space_sq_ft']))
data['Beds'] = (data['Beds'] - min(data['Beds'])) / (max(data['Beds']) - min(data['Beds']))
data['Baths'] = (data['Baths'] - min(data['Baths'])) / (max(data['Baths']) - min(data['Baths']))
data['Year'] = (data['Year'] - min(data['Year'])) / (max(data['Year']) - min(data['Year']))
print(data.head())
print(data.info())

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

layer1 = tf.keras.layers.Dense(units=5, input_shape = [4])
layer5 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([layer1,
                            tf.keras.layers.Dense(units=10),
                            tf.keras.layers.Dense(units=5),
                            layer5])

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.01),
    loss = 'mse'
)

x = train_df[['Living_Space_sq_ft', 'Beds', 'Baths', 'Year']].to_numpy()
y = train_df['SalePrice'].to_numpy()

print("COMIENZA EL ENTRENAMIENTO...")
historial = model.fit(x, y, epochs=200, verbose=True)
print("TERMINA EL ENTRENAMIENTO...")

plt.xlabel("# Epoca")
plt.ylabel("Loss magnitude")
plt.plot(historial.history["loss"])

prediction = model.predict(test_df[['Living_Space_sq_ft', 'Beds', 'Baths', 'Year']].to_numpy()).flatten()
realPrice = test_df['SalePrice'].to_numpy()

result = pd.DataFrame({'Prediction': prediction, 'Real price': realPrice})

diff = np.mean(abs( (realPrice - prediction)/realPrice ))

print(layer1.get_weights())
print(result)
print("Hay un MAPE de " + str(diff) + "%")
plt.show()