import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Housing/housePrices.csv")
print(data.head())
print(data.info())

train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

layer1 = tf.keras.layers.Dense(units=5, input_shape = [4])
layer3 = tf.keras.layers.Dense(units=1
                               )

model = tf.keras.Sequential([layer1, layer3])

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.01),
    loss = 'huber'
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

print("Hay un MAPE de " + str(diff) + "%")
print(result)
plt.show()