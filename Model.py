import keras_tuner as kt
import math
from keras import Sequential, Input
from keras.src.layers import Dense
import tensorflow as tf
from keras_tuner.src.backend import keras

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
        self.bayesian_opt_tuner = None
        self.best_hyperparameters = None
        self.metrics = []
        self.model = None

    def create_and_use_bayesian_opt_tuner(self):

        ####PRIMERA VUELTA####
        print("Entrada vuelta 1")
        #Se deciden numero de capas ocultas y una primera aprox de lr
        self.bayesian_opt_tuner = kt.BayesianOptimization(
            self.create_first_model_for_fine_tuning, objective="accuracy", max_trials=5, overwrite=True,
            directory='directorio_pruebas_bayesianTuner', project_name='Primer fine-tuning'
        )

        #deja en los atributos de la clase los resultados del fine tuning
        self.search_bayesian_opt_tuner()

        #asignamos resultados de la primera vuelta:
        self.assign_num_hidden_layers_to_model()
        self.assign_lr_to_model()

        ####SEGUNDA VUELTA###
        print("Entrada vuelta 2")
        #Se deciden numero de neuronas por capa y nueva aprox de lr

        if self.X_train.shape[1] >= 10:

            self.bayesian_opt_tuner = kt.BayesianOptimization(
                self.create_second_model_for_fine_tuning, objective="accuracy", max_trials=5, overwrite=True,
                directory='directorio_pruebas_bayesianTuner', project_name='Segundo fine-tuning'
            )

            # deja en los atributos de la clase los resultados del fine tuning
            self.search_bayesian_opt_tuner()

            # asignamos resultados de la segunda vuelta:
            self.assign_num_neurons_per_hidden_to_model()
            #self.assign_lr_to_model()   Hay que ver como hacerlo
        else:
            self.num_neurons_per_hidden = 10  # SI EL NUMERO DE FEATURES ES INFERIOR A 10, COGER 10 NEURONAS POR CAPA.


        ####TERCERA VUELTA####
        print("Entrada vuelta 3")
        self.bayesian_opt_tuner = kt.BayesianOptimization(
            self.create_third_model_for_fine_tuning, objective="accuracy", max_trials=5, overwrite=True,
            directory='directorio_pruebas_bayesianTuner', project_name='Tercer fine-tuning'
        )

        # deja en los atributos de la clase los resultados del fine tuning
        self.search_bayesian_opt_tuner()

        # asignamos resultados de la primera vuelta:
        self.assign_optimizer_to_model()

        #al fin, se construye el modelo final
        self.model = self.build_definitive_model()

    def search_bayesian_opt_tuner(self):
            self.bayesian_opt_tuner.search(self.X_train, self.y_train, epochs=10)
            self.best_hyperparameters = self.bayesian_opt_tuner.get_best_hyperparameters(num_trials=1)[0].values

    def assign_num_hidden_layers_to_model(self):
        self.num_hidden_layers = self.best_hyperparameters['num_hidden']

    def assign_lr_to_model(self):
        self.lr = self.best_hyperparameters['lr']

    def assign_optimizer_to_model(self):
        self.optimizer = self.best_hyperparameters['optimizer']

    def assign_num_neurons_per_hidden_to_model(self):
        self.num_neurons_per_hidden = self.best_hyperparameters['num_neurons_per_hidden']

    def build_definitive_model(self):

        model = Sequential()
        model.add(Input(self.num_neurons_input_layer))

        # Se añaden el resto de capas del modelo
        for i in range(self.num_hidden_layers):
            model.add(Dense(self.num_neurons_per_hidden, activation=self.hidden_activation_function[i]))

        # Se añade capa de salida. La función de activación corresponde al último
        model.add(Dense(self.num_neurons_output_layer, activation=self.output_activation_function))

        #Se vuelve a crear instancia de optimizador, ya que el anterior ya está modificado y no puede ser usado
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)

        # Compiling the model. Hace falta especificar la métrica accuracy para que el objeto history del model.fit contenga tal métrica
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    #se inicializan los hiperparámetros que son fijos, es decir, que no hay que hacer fine tuning de momento. Para organizar codigo en mas funciones
    def initialize_fixed_hiperparameters(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss = 'categorical_crossentropy'  # uso categorical_crossentropy cuando las etiquetas están codificadas con one-hot encoder. Si no usaría: sparse_categ_cross
        self.hidden_activation_function = ['relu', 'relu']  # Array: 1º valor -> 1 capa oculta ...
        self.output_activation_function = 'softmax'
        self.num_neurons_output_layer = self.y_train.shape[1]  # Depende de la estructura de y_train
        self.num_neurons_input_layer = (self.X_train.shape[1],)  # Depende de la estructura de X_train
        self.num_neurons_per_hidden = 10 #primero se halla numero de capas con este valor de neuronas por capa

    def build_arquitecture_for_fine_tuning(self):
        model = Sequential()
        model.add(Input(self.num_neurons_input_layer))

        # Se añaden el resto de capas del modelo
        for i in range(self.num_hidden_layers):
            model.add(Dense(self.num_neurons_per_hidden, activation=self.hidden_activation_function[i]))

        # Se añade capa de salida. La función de activación corresponde al último
        model.add(Dense(self.num_neurons_output_layer, activation=self.output_activation_function))

        # Compiling the model. Hace falta especificar la métrica accuracy para que el objeto history del model.fit contenga tal métrica
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    #Se decide optimizador
    def create_third_model_for_fine_tuning(self,hp):

        optimizerChoice = hp.Choice("optimizer",['adam', 'rmsprop', 'adamax'])
        #también se reentrena lr, ya que salia aviso de que si se exploraban mas valores (menores de 1e-5) podia ir mejor
        lr = hp.Float("lr",min_value= (self.lr / 100), max_value= (self.lr * 100))

        if optimizerChoice == 'adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        if optimizerChoice == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        if optimizerChoice == "adamax":
            self.optimizer = keras.optimizers.Adamax(learning_rate=lr)

        model = self.build_arquitecture_for_fine_tuning()
        return model

    #se deciden numero num_neuronas_por_capa y lr otra vez
    def create_second_model_for_fine_tuning(self,hp):
        if self.X_train.shape[1] < 10:
            self.num_neurons_per_hidden = 10  # SI EL NUMERO DE FEATURES ES INFERIOR A 10, COGER 10 NEURONAS POR CAPA.
        else:
            if self.X_train.shape[1] > 100:
                self.num_neurons_per_hidden = hp.Int("num_neurons_per_hidden", min_value=10,max_value=self.X_train.shape[1],sample='log')  # Si hay muchas features, se hace sample log para que coja valores que representen la gran variación de los posibles valores.
            else:
                self.num_neurons_per_hidden = hp.Int("num_neurons_per_hidden", min_value=10,max_value=self.X_train.shape[1])

        ########Hay que ver como hacer con la lr de nuevo
        model = self.build_arquitecture_for_fine_tuning()
        return model


    #se deciden num_capas y primera aprox de lr
    def create_first_model_for_fine_tuning(self,hp):

        self.num_hidden_layers = hp.Int("num_hidden", min_value=2, max_value=math.ceil(math.sqrt(self.X_train.shape[1])))  # Entre 2 - sqroot(nº features)

        self.lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling='log')

        #En la primera vuelta hay que inicializar los hiperparámetros con los que no se hará fine-tuning
        self.initialize_fixed_hiperparameters()

        model = self.build_arquitecture_for_fine_tuning()
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