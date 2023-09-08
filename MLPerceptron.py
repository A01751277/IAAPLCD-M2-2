# Actividad Retroalimentación Módulo 2
# Alejandro Somarriba Aguirre
# A01751277

# Se importan las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

# Se cargan los datos a partir del paquete de Scikit-Learn
from sklearn.datasets import load_digits

# Se guardan los datos en una variable
digits = load_digits()

# Se separan los datos en datos de entrada y salida
X = pd.DataFrame(digits.data, columns = digits.feature_names)
y = pd.DataFrame(digits.target)

# Se separan los datos aún más en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Se inicializa un objeto de un clasificador de perceptrón multi-capa
# Se indica que tiene 2 capas ocultas; una con 32 neuronas y otra con 10 neuronas
# Las funciones de activación para las capas ocultas son ReLU
# El algoritmo de optimización de parámetros es SDG (Gradiente Estocástico)
# Por último, se le indica que reserve un 10% de los datos de entrenamiento para validación
# Por default el algoritmo tiene una tasa de aprendizaje de 0.001, que se mantiene constante en el entrenamiento
# Del mismo modo, el entrenamiento por default corre por 200 iteraciones
clasificador = MLPClassifier(hidden_layer_sizes = (32, 10), activation = "relu", solver = "sgd", verbose = True, validation_fraction = 0.1)

# Se imprime información más detallada del modelo
print("Características del modelo:")
print(clasificador.get_params())

# Se entrena el modelo usando el método .fit()
# Como en el modelo se indicó "verbose = True", se imprime el valor de loss por cada iteración
print("\nComenzando Entrenamiento...\n")
clasificador.fit(X_train, y_train)

# La función de activación para la capa de salida es la función Softmax
print("\nFunción de la capa de salida:")
print(clasificador.out_activation_)

# Al terminar el entrenamiento, se imprimen los valores de Accuracy del modelo para los datos de entrenamiento y los de prueba
print("\nAccuracy con datos de Entrenamiento: " + str(clasificador.score(X_train, y_train)))
print("\nAccuracy con datos de Prueba: " + str(clasificador.score(X_test, y_test)))

# Después de imprimir los valores de Accuracy, se muestra una gráfica de cómo cambia el loss con cada iteración
plt.plot(clasificador.loss_curve_)
plt.title("Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Luego, se guardan los resultados de las predicciones realizadas a partir de los datos de prueba
# La siguiente línea comentada puede usarse para guardar las probabilidades de cada clase
# y_pred_prob = clasificador.predict_proba(X_test)
y_pred = clasificador.predict(X_test)

# Antes de terminar, se imprime un resumen que incluye varias métricas de evaluación
print(classification_report(y_test, y_pred))

# Finalmente, se obtiene una matriz de confusión con los resultados
conf_m = confusion_matrix(y_test, y_pred)

# Se guarda la matriz como un DataFrame para poder mostrarla después
df_conf_m = pd.DataFrame(columns = digits.target_names, index = digits.target_names, data = conf_m)

# Se configuran los parámetros para mostrar la matriz de confusión con colores
f,ax = plt.subplots(figsize=(6,6))

sns.heatmap(df_conf_m, annot=True,cmap="Greens", fmt= '.0f',
            ax=ax,linewidths = 5, cbar = False, annot_kws={"size": 14})
plt.xlabel("Predicted Label")
plt.xticks(size = 10)
plt.yticks(size = 10, rotation = 0)
plt.ylabel("True Label")
plt.title("Confusion Matrix", size = 10)
plt.show()