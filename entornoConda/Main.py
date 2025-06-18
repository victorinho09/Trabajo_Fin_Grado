from ucimlrepo import fetch_ucirepo
from Model import Model
from entrenamiento import train_and_evaluate
from preprocesamiento import preprocess_dataset

#data_iris = fetch_ucirepo(id=53).data #clasificacion
#print("Iris dataset cargado")
#train_and_evaluate(data_iris,"iris",user_num_epochs=30)

data_mushroom = fetch_ucirepo(id=73).data #clasificacion
print("Mushroom dataset cargado")
#train_and_evaluate(data_mushroom,500, "logs/fit/mushroom/500batches",16,"mushroom")

# El user si quiere preprocesar dataset le damos la funcion para que lo haga
X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = preprocess_dataset(data_mushroom,"mushroom")
model = Model(X_train_scaled, y_train_encoded,"directorio logs",user_max_trials=3,user_min_num_hidden_layers=6,user_max_num_hidden_layers=6,user_min_num_neurons_per_hidden=3,user_max_num_neurons_per_hidden=3)
model.autotune()
model.train()
loss,precision=model.evaluate(X_test_scaled, y_test_encoded)
hiperparams_and_params = model.get_final_hyperparams_and_params()
# El user deberá pasar el dataset a la inicializacion del modelo



# data_heart_disease = fetch_ucirepo(id=45).data #clasificacion
# print("Heart_disease dataset cargado")

# train_and_evaluate(data_heart_disease, "heart_disease")

# data_adult = fetch_ucirepo(id=2).data #clasificacion
# print("Adult dataset cargado")
# train_and_evaluate(data_adult,500, "logs/fit/adult/500batches",16,"adult")

#data_breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17).data #clasificacion
#print("Breast_cancer dataset cargado")
#train_and_evaluate(data_breast_cancer_wisconsin_diagnostic,500, "logs/fit/breast_cancer_wisconsin_diagnostic/500batches",16,"breast_cancer_wisconsin_diagnostic")

# data_bank_marketing = fetch_ucirepo(id=222).data #clasificacion
# print("Bank_marketing dataset cargado")
# train_and_evaluate(data_bank_marketing,500, "logs/fit/bank_marketing/500batches",16,"bank_marketing")


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

# data_phiusiil_phishing_url_website = fetch_ucirepo(id=967).data #clasificacion.
