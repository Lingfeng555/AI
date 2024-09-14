#import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint

def convertFarenheit (celcius) -> float : return (celcius*1.8 + 32)

def main():
    celcius = []
    farenheit = []
    for i in range (55):
        celcius.append(randint(-100,100))
        farenheit.append(convertFarenheit(celcius[i]))
    data = pd.DataFrame({'Celcius': celcius, 'Farenheit': farenheit})
    print(data.head())

    layer = tf.keras.layers.Dense(units =1, input_shape=[1])
    model = tf.keras.Sequential([layer])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.2),
        loss = 'mean_squared_error'
    )

    print("trainning")
    historial = model.fit(data['Celcius'].to_numpy(), data['Farenheit'].to_numpy(), epochs=300, verbose=True)
    print("trainned")

    plt.xlabel("# Epoca")
    plt.ylabel("Loss magnitude")
    plt.plot(historial.history["loss"])

    plt.show()

    prediction = model.predict(np.array([100.0]))
    print("Prediction: " + str(prediction[0][0]) + " | Real value: " + str(convertFarenheit(100.0)))

    print ("Values: " )
    print(layer.get_weights())

    return

if __name__ == '__main__': main()
