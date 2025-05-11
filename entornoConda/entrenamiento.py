from Model import Model
from funciones_preprocesamiento import preprocess_data


def train_and_evaluate(dataset,nombre_fichero_info_dataset, num_batches=None, log_dir = "logs/fit/default_dataset_name", batch_size= 16,user_num_epochs= None, user_max_trials=None):
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = preprocess_data(dataset,nombre_fichero_info_dataset)

    model = Model(X_train_scaled, y_train_encoded,log_dir,batch_size,num_batches,user_num_epochs=user_num_epochs,user_max_trials=user_max_trials)
    model.autotune()
    model.train()
    loss,precision=model.evaluate(X_test_scaled, y_test_encoded)
    lista_hiperparams = model.get_final_hyperparams()
    return [nombre_fichero_info_dataset,user_num_epochs,user_max_trials,loss,precision] + lista_hiperparams