import keras_tuner
from keras_tuner.src import keras_tuner_export

from Model import Model
from funciones_preprocesamiento import preprocess_data


def train_and_evaluate(dataset, num_batches, log_dir, batch_size):
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = preprocess_data(dataset)

    model = Model(X_train_scaled, y_train_encoded)
    model.create_and_use_bayesian_opt_tuner()
    model.train(num_batches,batch_size,log_dir)
    model.evaluate(X_test_scaled, y_test_encoded)