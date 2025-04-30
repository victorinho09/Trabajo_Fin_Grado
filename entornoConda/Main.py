
from ucimlrepo import fetch_ucirepo

from entrenamiento import train_and_evaluate

data_iris = fetch_ucirepo(id=53).data #clasificacion
print("Iris dataset cargado")
train_and_evaluate(data_iris,50, "logs/fit/iris/500batches",16,"iris")

#ata_heart_disease = fetch_ucirepo(id=45).data #clasificacion
#print("Heart_disease dataset cargado")
#train_and_evaluate(data_heart_disease,500, "logs/fit/heart_disease/500batches",16,"heart_disease")

# data_adult = fetch_ucirepo(id=2).data #clasificacion
# print("Adult dataset cargado")
# train_and_evaluate(data_adult,500, "logs/fit/adult/500batches",16,"adult")

#data_breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17).data #clasificacion
#print("Breast_cancer dataset cargado")
#train_and_evaluate(data_breast_cancer_wisconsin_diagnostic,500, "logs/fit/breast_cancer_wisconsin_diagnostic/500batches",16,"breast_cancer_wisconsin_diagnostic")

# data_bank_marketing = fetch_ucirepo(id=222).data #clasificacion
# print("Bank_marketing dataset cargado")
# train_and_evaluate(data_bank_marketing,500, "logs/fit/bank_marketing/500batches",16,"bank_marketing")

# data_mushroom = fetch_ucirepo(id=73).data #clasificacion
# print("Mushroom dataset cargado")
# train_and_evaluate(data_mushroom,500, "logs/fit/mushroom/500batches",16,"mushroom")


# data_default_payment = fetch_ucirepo(id=350).data #clasificacion
# print("Default_payment dataset cargado")
# train_and_evaluate(data_default_payment,500, "logs/fit/default_payment/500batches",16,"default_payment")


# data_car_evaluation = fetch_ucirepo(id=19).data #clasificacion
# print("Car_evaluation dataset cargado")
# train_and_evaluate(data_car_evaluation,500, "logs/fit/car_evaluation/500batches",16,"car_evaluation")

# data_dry_bean = fetch_ucirepo(id=602).data #clasificacion
# print("Dry_bean dataset cargado")
# train_and_evaluate(data_dry_bean,500, "logs/fit/dry_bean/500batches",16,"dry_bean")

# data_magic_gamma_telescope = fetch_ucirepo(id=159).data #clasificacion
# print("Magic_gamma_telescope dataset cargado")
# train_and_evaluate(data_magic_gamma_telescope,500, "logs/fit/magic_gamma_telescope/500batches",16,"magic_gamma_telescope")

# data_spambase = fetch_ucirepo(id=94).data #clasificacion
# print("Spambase dataset cargado")
# train_and_evaluate(data_spambase,500, "logs/fit/spambase/500batches",16,"spambase")

# data_census_income = fetch_ucirepo(id=20).data #clasificacion
# print("Census_income dataset cargado")
# train_and_evaluate(data_census_income,500, "logs/fit/census_income/500batches",16,"census_income")

# data_phiusiil_phishing_url_website = fetch_ucirepo(id=967).data #clasificacion. --> NO CABE EN RAM, DA ERROR
# data_bike_sharing = fetch_ucirepo(id=275).data #regresion --> Este dataset funciona muy mal. ¿Mejorará con otra arquitectura?
# data_real_estate_valuation = fetch_ucirepo(id=477).data #regresion --> Problema: Demasiado numero de clases objetivo, y muy pocas instancias, por lo que no da suficiente para que stratify funcione en testset, no consigue meter instancias en train y test set de igual manera
# data_communities_and_crime = fetch_ucirepo(id=183).data #regresion --> Funciona muy mal tmb, muchas clases objetivo
# data_parkinsons_telemonitoring = fetch_ucirepo(id=189).data #regresion -->No coge datos, no contiene nada