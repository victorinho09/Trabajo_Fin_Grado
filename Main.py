
from ucimlrepo import fetch_ucirepo

from entrenamiento import train_and_evaluate

data_iris = fetch_ucirepo(id=53).data #clasificacion
print("Iris dataset cargado")

'''
data_heart_disease = fetch_ucirepo(id=45).data #clasificacion
print("Heart_disease dataset cargado")
data_adult = fetch_ucirepo(id=2).data #clasificacion
print("Adult dataset cargado")
data_breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17).data #clasificacion
print("Breast_cancer dataset cargado")
data_bank_marketing = fetch_ucirepo(id=222).data #clasificacion
print("Bank_marketing dataset cargado")
data_diabetes = fetch_ucirepo(id=296).data #clasificacion
print("Diabetes dataset cargado")
data_mushroom = fetch_ucirepo(id=73).data #clasificacion
print("Mushroom dataset cargado")
data_default_payment = fetch_ucirepo(id=350).data #clasificacion
print("Default_payment dataset cargado")
data_car_evaluation = fetch_ucirepo(id=19).data #clasificacion
print("Car_evaluation dataset cargado")
data_dry_bean = fetch_ucirepo(id=602).data #clasificacion
print("Dry_bean dataset cargado")
data_magic_gamma_telescope = fetch_ucirepo(id=159).data #clasificacion
print("Magic_gamma_telescope dataset cargado")
data_spambase = fetch_ucirepo(id=94).data #clasificacion
print("Spambase dataset cargado")
data_census_income = fetch_ucirepo(id=20).data #clasificacion
print("Census_income dataset cargado")
'''
'''
data_phiusiil_phishing_url_website = fetch_ucirepo(id=967).data #clasificacion. --> NO CABE EN RAM, DA ERROR
data_bike_sharing = fetch_ucirepo(id=275).data #regresion --> Este dataset funciona muy mal. ¿Mejorará con otra arquitectura?
data_real_estate_valuation = fetch_ucirepo(id=477).data #regresion --> Problema: Demasiado numero de clases objetivo, y muy pocas instancias, por lo que no da suficiente para que stratify funcione en testset, no consigue meter instancias en train y test set de igual manera
data_communities_and_crime = fetch_ucirepo(id=183).data #regresion --> Funciona muy mal tmb, muchas clases objetivo
data_parkinsons_telemonitoring = fetch_ucirepo(id=189).data #regresion -->No coge datos, no contiene nada
'''

#numero de neuronas rango (capa inicial) : raiz del numero de features - numero de features
#numero capas:


train_and_evaluate(data_iris,500, "logs/fit/iris/500batches",16)
'''
train_and_evaluate(data_heart_disease,500, "logs/fit/heart_disease/500batches",16)
train_and_evaluate(data_breast_cancer_wisconsin_diagnostic,500, "logs/fit/breast_cancer_wisconsin_diagnostic/500batches",16)
train_and_evaluate(data_bank_marketing,500, "logs/fit/bank_marketing/500batches",16)
train_and_evaluate(data_diabetes,500, "logs/fit/diabetes/500batches",16)
train_and_evaluate(data_adult,500, "logs/fit/adult/500batches",16)
train_and_evaluate(data_mushroom,500, "logs/fit/mushroom/500batches",16)
train_and_evaluate(data_default_payment,500, "logs/fit/default_payment/500batches",16)
train_and_evaluate(data_dry_bean,500, "logs/fit/dry_bean/500batches",16)
train_and_evaluate(data_spambase,500, "logs/fit/spambase/500batches",16)
train_and_evaluate(data_magic_gamma_telescope,500, "logs/fit/magic_gamma_telescope/500batches",16)
train_and_evaluate(data_car_evaluation,500, "logs/fit/car_evaluation/500batches",16)
train_and_evaluate(data_census_income,500, "logs/fit/census_income/500batches",16)
'''

# %tensorboard --logdir logs/fit --port=6007