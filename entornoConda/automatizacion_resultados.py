from entrenamiento import train_and_evaluate
from ucimlrepo import fetch_ucirepo
from funciones_auxiliares import pintar_resultado_en_fichero


def ejecutar_prueba_interna(data,nombre_fichero_dataset,user_num_epochs,user_max_trials):
    #Probar con distintos numero de epocas. Para cada numero de epocas probar con distintos numeros de trials
    datos_entrenamiento = train_and_evaluate(data,nombre_fichero_dataset,user_num_epochs=user_num_epochs,user_max_trials=user_max_trials)
    pintar_resultado_en_fichero(datos_entrenamiento)


nombre_fichero_dataset = "iris"
data_iris = fetch_ucirepo(id=53).data  # clasificacion

ejecutar_prueba_interna(data_iris, nombre_fichero_dataset, 100, 70)
ejecutar_prueba_interna(data_iris, nombre_fichero_dataset, 100, 70)
ejecutar_prueba_interna(data_iris, nombre_fichero_dataset, 100, 60)
ejecutar_prueba_interna(data_iris, nombre_fichero_dataset, 100, 60)


