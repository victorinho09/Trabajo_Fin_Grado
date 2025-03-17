from keras import Sequential, Input
from keras.src.layers import Dense
import tensorflow as tf
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics

from EpochCumulativeLogger import EpochCumulativeLogger
from GlobalBatchLogger import GlobalBatchLogger
from funciones import get_num_epochs_train, dividir_array


class  Model():
    def __init__(self,X_train,y_train):

        #Para ser usados luego en la estructura de la capa inicial, final y para el entrenamiento del modelo
        self.X_train = X_train
        self.y_train = y_train

        #No obtendrá valor hasta que se entrene el modelo
        self.history = None

        self.num_epochs = None
        self.num_batches_per_epoch = None

        self.model = Sequential()

        #Se inicializan ciertas variables con valores por defecto, ya que no hay más referencias en la primera instancia

        self.num_dense_layers = 2

        #Array: 1º valor -> 1º capa ...
        self.num_neurons_per_dense_layer = [64,32]

        #Depende de la estructura de y_train
        self.num_neurons_output_layer = y_train.shape[1]

        #Depende de la estructura de X_train
        self.num_neurons_input_layer = (X_train.shape[1],)

        #De momento, se usa solo el optimizer
        self.learning_rate = None
        self.optimizer = 'adam'

        self.loss = 'categorical_crossentropy'  # uso categorical_crossentropy cuando las etiquetas están codificadas con one-hot encoder. Si no usaría: sparse_categ_cross

        #Array: 1º valor -> 1 capa oculta ...
        self.dense_layers_activation_function = ['relu','relu']
        self.output_layer_activation_function = 'softmax'
        self.metrics = []

        self.create_model()


    def create_model(self):

        self.model.add(Input(self.num_neurons_input_layer))

        #Se añaden el resto de capas del modelo
        for i,num_neurons_layer in enumerate(self.num_neurons_per_dense_layer):
            self.model.add(Dense(num_neurons_layer, activation= self.dense_layers_activation_function[i]))

        #Se añade capa de salida. La función de activación corresponde al último
        self.model.add(Dense(self.num_neurons_output_layer, activation= self.output_layer_activation_function ))

        # Compiling the model. Hace falta especificar la métrica accuracy para que el objeto history del model.fit contenga tal métrica
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

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

