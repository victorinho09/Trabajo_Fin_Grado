from entrenamiento import train_and_evaluate
from ucimlrepo import fetch_ucirepo
from funciones_auxiliares import pintar_resultado_en_fichero, borrar_fichero_pruebas_internas

nombre_fichero_dataset = "iris"
data_iris = fetch_ucirepo(id=53).data  # clasificacion

#Al principio de la prueba se borra el fichero que contiene infor de un dataset de otra prueba anterior
borrar_fichero_pruebas_internas(nombre_fichero_dataset)

#Probar con distintos numero de epocas. Para cada numero de epocas probar con distintos numeros de trials
datos_entrenamiento = train_and_evaluate(data_iris,nombre_fichero_dataset,user_num_epochs=30,user_max_trials=10)
pintar_resultado_en_fichero(datos_entrenamiento)
