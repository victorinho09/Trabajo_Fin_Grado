from entrenamiento import train_and_evaluate
from ucimlrepo import fetch_ucirepo

data_iris = fetch_ucirepo(id=53).data  # clasificacion
#Probar con distintos numero de epocas. Para cada numero de epocas probar con distintos numeros de trials
lista_datos_entrenamiento = train_and_evaluate(data_iris,"iris",user_num_epochs=30,user_max_trials=10)
print("###########################################")
print(lista_datos_entrenamiento)
