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

# data = fetch_ucirepo(id=350).data #clasificacion
# nombre_fichero_dataset = "default_payment"

# data= fetch_ucirepo(id=19).data #clasificacion
# nombre_fichero_dataset = "car_evaluation"

# data = fetch_ucirepo(id=602).data #clasificacion
# nombre_fichero_dataset = "dry_bean"

# data = fetch_ucirepo(id=159).data #clasificacion
# nombre_fichero_dataset = "magic_gamma_telescope"

# data = fetch_ucirepo(id=94).data #clasificacion
# nombre_fichero_dataset = "spambase"

# data = fetch_ucirepo(id=20).data #clasificacion
# nombre_fichero_dataset = "census_income"

# data = fetch_ucirepo(id=967).data #clasificacion.
# nombre_fichero_dataset = "phiusiil_phising_url"

for i in range(10):
    ejecutar_prueba_interna(data, nombre_fichero_dataset, 5, 10)

# for i in range(5):
#     ejecutar_prueba_interna(data, nombre_fichero_dataset, 5, 10)
#
# for i in range(5):
#     ejecutar_prueba_interna(data, nombre_fichero_dataset, 5, 20)
#
# for i in range(5):
#     ejecutar_prueba_interna(data, nombre_fichero_dataset, 10, 10)


