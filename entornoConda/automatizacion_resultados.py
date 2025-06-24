from entrenamiento import train_and_evaluate
from ucimlrepo import fetch_ucirepo
from funciones_auxiliares import pintar_resultado_en_fichero
import kagglehub
import pandas as pd
from pathlib import Path


def ejecutar_prueba_interna(data,nombre_fichero_dataset,user_num_epochs,user_max_trials,nombre_clase_objetivo):
    #Probar con distintos numero de epocas. Para cada numero de epocas probar con distintos numeros de trials
    datos_entrenamiento = train_and_evaluate(data,nombre_fichero_dataset,nombre_clase_objetivo=nombre_clase_objetivo,user_num_epochs=user_num_epochs,user_max_trials=user_max_trials)
    pintar_resultado_en_fichero(datos_entrenamiento)


# nombre_fichero_dataset = "heart_disease"
# data = fetch_ucirepo(id=45).data #clasificacion

# data = fetch_ucirepo(id=73).data #clasificacion
# nombre_fichero_dataset = "mushroom"

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

# ruta_fichero_kaggle = "elikplim/car-evaluation-data-set"
# ruta_fichero_csv_interno = "car_evaluation.csv"
# nombre_fichero_dataset = "car_evaluation"
# nombre_clase_objetivo = "unacc"

# ruta_fichero_kaggle = "sansuthi/dry-bean-dataset"
# ruta_fichero_csv_interno = "Dry_Bean.csv"
# nombre_fichero_dataset = "dry_bean"
# nombre_clase_objetivo = "Class"

# ruta_fichero_kaggle = "uciml/adult-census-income"
# ruta_fichero_csv_interno = "adult.csv"
# nombre_fichero_dataset = "census_income"
# nombre_clase_objetivo = "income"


# ruta_fichero_kaggle = "colormap/spambase"
# ruta_fichero_csv_interno = "spambase.csv"
# nombre_fichero_dataset = "spambase"
# nombre_clase_objetivo = "spam"

# ruta_fichero_kaggle = "abhinand05/magic-gamma-telescope-dataset"
# ruta_fichero_csv_interno = "telescope_data.csv"
# nombre_fichero_dataset = "magic_gamma_telescope"
# nombre_clase_objetivo = "class"

ruta_fichero_kaggle = "uciml/default-of-credit-card-clients-dataset"
ruta_fichero_csv_interno = "UCI_Credit_Card.csv"
nombre_fichero_dataset = "default_payment"
nombre_clase_objetivo = "default.payment.next.month"


dataset_dir = kagglehub.dataset_download(ruta_fichero_kaggle)
dataset_path = Path(dataset_dir)
data = pd.read_csv(dataset_path / ruta_fichero_csv_interno)

#Verifica que esté bien
#print(data.head())

for i in range(3):
    ejecutar_prueba_interna(data, nombre_fichero_dataset,nombre_clase_objetivo=nombre_clase_objetivo, user_num_epochs=5, user_max_trials=1)
for i in range(3):
    ejecutar_prueba_interna(data, nombre_fichero_dataset,nombre_clase_objetivo=nombre_clase_objetivo, user_num_epochs=5,user_max_trials=2)
for i in range(3):
    ejecutar_prueba_interna(data, nombre_fichero_dataset,nombre_clase_objetivo=nombre_clase_objetivo, user_num_epochs=5,user_max_trials=4)
for i in range(3):
    ejecutar_prueba_interna(data, nombre_fichero_dataset,nombre_clase_objetivo=nombre_clase_objetivo, user_num_epochs=5,user_max_trials=8)
for i in range(3):
    ejecutar_prueba_interna(data, nombre_fichero_dataset,nombre_clase_objetivo=nombre_clase_objetivo, user_num_epochs=5,user_max_trials=16)
for i in range(3):
    ejecutar_prueba_interna(data, nombre_fichero_dataset,nombre_clase_objetivo=nombre_clase_objetivo, user_num_epochs=5,user_max_trials=32)
