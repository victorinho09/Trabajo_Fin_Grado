from Model import Model
from preprocesamiento import preprocess_dataset


def train_and_evaluate(dataset,nombre_fichero_info_dataset, num_batches=None, log_dir = "logs/fit/default_dataset_name", batch_size= 16,user_num_epochs= None, user_max_trials=None):
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = preprocess_dataset(dataset,nombre_fichero_info_dataset)

    model = Model(X_train_scaled, y_train_encoded,log_dir,batch_size,num_batches,user_num_epochs=user_num_epochs,user_max_trials=user_max_trials)
    model.autotune()
    model.train()
    loss,precision=model.evaluate(X_test_scaled, y_test_encoded)
    hiperparams_and_params = model.get_final_hyperparams_and_params()

    return {
        "nombre_dataset": nombre_fichero_info_dataset,
        "loss": loss,
        "precision": precision,
        "lr" : hiperparams_and_params[0],
        "optimizador" : hiperparams_and_params[1],
        "funcion_activacion" : hiperparams_and_params[2],
        "numero_neuronas_por_capa_oculta" : hiperparams_and_params[3],
        "numero_capas_ocultas" : hiperparams_and_params[4],
        "numero_epocas" : hiperparams_and_params[5],
        "max_trials" : hiperparams_and_params[6],

    }