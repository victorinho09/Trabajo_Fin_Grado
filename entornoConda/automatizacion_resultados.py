from entrenamiento import train_and_evaluate
from ucimlrepo import fetch_ucirepo
from funciones_auxiliares import pintar_resultado_en_fichero


def ejecutar_prueba_interna(data,nombre_fichero_dataset,user_num_epochs,user_max_trials):
    #Probar con distintos numero de epocas. Para cada numero de epocas probar con distintos numeros de trials
    datos_entrenamiento = train_and_evaluate(data,nombre_fichero_dataset,user_num_epochs=user_num_epochs,user_max_trials=user_max_trials)
    pintar_resultado_en_fichero(datos_entrenamiento)


# nombre_fichero_dataset = "iris"
# data =fetch_ucirepo(id=53).data  # clasificacion

# nombre_fichero_dataset = "heart_disease"
# data = fetch_ucirepo(id=45).data #clasificacion

data = fetch_ucirepo(id=73).data #clasificacion
nombre_fichero_dataset = "mushroom"

# data = fetch_ucirepo(id=17).data #clasificacion
# nombre_fichero_dataset = "breast_cancer_wisconsin_diagnostic"

# data = fetch_ucirepo(id=222).data #clasificacion
# nombre_fichero_dataset = "bank_marketing"

for i in range(6):
    ejecutar_prueba_interna(data, nombre_fichero_dataset, 5, 3)

for i in range(6):
    ejecutar_prueba_interna(data, nombre_fichero_dataset, 5, 10)

for i in range(6):
    ejecutar_prueba_interna(data, nombre_fichero_dataset, 5, 20)

for i in range(6):
    ejecutar_prueba_interna(data, nombre_fichero_dataset, 10, 10)


