import keras_tuner as kt
import math
from keras import Sequential, Input
from keras.src.layers import Dense
import tensorflow as tf

from EpochCumulativeLogger import EpochCumulativeLogger
from GlobalBatchLogger import GlobalBatchLogger
from funciones_auxiliares import get_num_epochs_train, dividir_array


class  Model():
    def __init__(self,X_train,y_train):

        self.X_train = X_train #Para ser usados luego en la estructura de la capa inicial, final y para el entrenamiento del modelo
        self.y_train = y_train
        self.history = None #No obtendrá valor hasta que se entrene el modelo
        self.num_epochs = None
        self.num_batches_per_epoch = None
        self.num_hidden_layers = None
        self.num_neurons_per_hidden = None
        self.lr = None
        self.optimizer = None
        self.loss = None
        self.hidden_activation_function = None
        self.output_activation_function = None
        self.num_neurons_output_layer = None
        self.num_neurons_input_layer = None
        self.metrics = []

        random_search_tuner = kt.RandomSearch(
            self.create_model, objective="accuracy", max_trials= 5, overwrite=True,
            directory='directorio_pruebas_rndomsearch',project_name='mi_rndomsearch'
        )
        random_search_tuner.search(self.X_train,self.y_train, epochs=10)

        best_model = random_search_tuner.get_best_hyperparameters(num_trials=1)
        best_hyperparameters = best_model[0].values
        print("Mejores valores: ",best_hyperparameters)

    def create_model(self,hp):

        self.num_hidden_layers = hp.Int("num_hidden",min_value=2,max_value=math.sqrt(self.X_train.shape[1])) #Entre 2 - sqroot(nº features)
        self.num_neurons_per_hidden = [64,32] #Array: 1º valor -> 1º capa ...
        self.lr = hp.Float("lr",min_value=1e-5,max_value=1e-2,sampling= 'log')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
        self.loss= 'categorical_crossentropy' # uso categorical_crossentropy cuando las etiquetas están codificadas con one-hot encoder. Si no usaría: sparse_categ_cross
        self.hidden_activation_function = ['relu','relu'] #Array: 1º valor -> 1 capa oculta ...
        self.output_activation_function = 'softmax'
        self.num_neurons_output_layer = self.y_train.shape[1]  # Depende de la estructura de y_train
        self.num_neurons_input_layer = (self.X_train.shape[1],)  # Depende de la estructura de X_train

        model = Sequential()
        model.add(Input(self.num_neurons_input_layer))

        #Se añaden el resto de capas del modelo
        for i in range(self.num_hidden_layers):
            model.add(Dense(self.num_neurons_per_hidden[i], activation= self.hidden_activation_function[i]))

        #Se añade capa de salida. La función de activación corresponde al último
        model.add(Dense(self.num_neurons_output_layer, activation= self.output_activation_function ))

        # Compiling the model. Hace falta especificar la métrica accuracy para que el objeto history del model.fit contenga tal métrica
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    def train(self,num_batches,batch_size,log_dir):

        global_batch_logger = GlobalBatchLogger(log_dir)
        global_epoch_logger = EpochCumulativeLogger(log_dir)
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        )

        validation_split_value = 0.2
        # Hay que tener en cuenta que el validation split es 0.2, por lo que realmente se usa solo el 80% del dataset de entrenamiento
        self.num_epochs, self.num_batches_per_epoch = get_num_epochs_train(batch_size, self.X_train, num_batches,validation_split_value)

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.2,
            # Se usa validation split para que automáticamente divida el train set. Con validation data hay que separarlo manualmente.
            shuffle=True,  # Para que baraje los datos antes de la división del val set
            epochs= self.num_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[tb_callback, global_epoch_logger, global_batch_logger]
        )

        #Obtenemos las metricas
        self.set_metrics(global_batch_logger)

    def set_metrics(self,global_batch_logger):

        #print(self.history.history.keys())
        metric_val_accuracy = self.history.history["val_accuracy"]
        metric_val_loss = self.history.history["val_loss"]
        metric_loss_per_batch = dividir_array(global_batch_logger.batch_loss_acum, self.num_batches_per_epoch)
        metric_accuracy_per_batch = dividir_array(global_batch_logger.batch_accuracy_acum, self.num_batches_per_epoch)

        # print("Número de elementos en val_accuracy: ", len(metric_val_accuracy))
        # print("Número de elementos en val_loss: ", len(metric_val_loss))
        # print("Número de elementos/listas en loss_per_batch: ", len(metric_loss_per_batch))
        # print("Número de elementos/listas en accuracy_per_batch: ", len(metric_accuracy_per_batch))

        # print("Número de batches por época: ",num_batches_per_epoch)
        # print("metric_loss_per_batch: ", metric_loss_per_batch)
        # print("metric_accuracy_per_batch",metric_accuracy_per_batch)

        for epoch in range(self.num_epochs):
            info_epoch = [metric_accuracy_per_batch[epoch], metric_loss_per_batch[epoch], metric_val_accuracy[epoch],
                          metric_val_loss[epoch]]
            self.metrics.append(info_epoch)
            print(info_epoch)
            # print(epoch)

    def evaluate(self,X_test_scaled, y_test_encoded):
        self.model.evaluate(X_test_scaled, y_test_encoded)

'''
    def search_best_params(self):

        param_dist = {
            'num_dense_layers': self.num_dense_layers,
            'lr': self.learning_rate,
        }
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions= param_dist,
            cv=2,
            scoring="f1_macro"
        )

        random_search.fit(self.X_train,self.y_train)
        print(random_search.best_params_)
'''