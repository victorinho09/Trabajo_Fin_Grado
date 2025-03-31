import keras_tuner as kt
import math
from keras import Sequential, Input
from keras.src.layers import Dense
import tensorflow as tf
from keras_tuner.src.backend import keras
from sklearn.model_selection import train_test_split

from EpochCumulativeLogger import EpochCumulativeLogger
from GlobalBatchLogger import GlobalBatchLogger
from funciones_auxiliares import get_num_epochs_train, dividir_array


class  Model():
    def __init__(self,X_train,y_train,log_dir,batch_size,num_batches,X_val=None,y_val=None):

        global_batch_logger = GlobalBatchLogger(log_dir)
        global_epoch_logger = EpochCumulativeLogger(log_dir)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

        #Se coje los datos de validación por paramétro si existen, si no se crean del train set
        if  (X_val is None) or (y_val is None) :
            #hacer split del train
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=0.3) # Se genera el conjunto de validacion y se ponen como argumentos todos los datasets necesarios
        else:
            self.X_val= X_val
            self.y_val= y_val
            self.X_train = X_train
            self.y_train = y_train
        self.validation_data = (self.X_val,self.y_val)

        # atributos de parametros pasados al fit method de funcion train
        self.validation_split = 0.2  # Se usa validation split para que automáticamente divida el train set. Con validation data hay que separarlo manualmente.
        self.shuffle = True  # Para que baraje los datos antes de la división del val set
        self.batch_size = batch_size
        self.verbose = 0
        self.log_dir = log_dir
        self.callbacks = [tb_callback, global_epoch_logger, global_batch_logger]

        # atributos de parametros pasados al tuner
        self.max_trials = 5
        self.objective = "val_accuracy"
        self.overwrite = True
        self.directory = "bayesian_tuner"

        #atributo del metodo search del tuner
        self.num_epochs_tuner = 5

        self.num_batches = num_batches
        self.num_epochs, self.num_batches_per_epoch = get_num_epochs_train(self.batch_size, self.X_train, self.num_batches,self.validation_split)

        self.history = None #No obtendrá valor hasta que se entrene el modelo
        self.num_hidden_layers = None
        self.lr = None
        self.num_neurons_per_hidden = 10
        self.optimizers_list = ['adam', 'rmsprop', 'adamax']
        self.optimizer = None
        self.loss = 'categorical_crossentropy' # uso categorical_crossentropy cuando las etiquetas están codificadas con one-hot encoder. Si no usaría: sparse_categ_cross
        self.hidden_activation_function = 'relu'  # misma funcion de activacion para todas las capas. NO merece la pena tener 1 distinta por cada capa. Se incrementa demasiado número de hiperparaemetros que tunear
        self.output_activation_function = 'softmax'
        self.num_neurons_output_layer = self.y_train.shape[1]  # Depende de la estructura de y_train
        self.num_neurons_input_layer = (self.X_train.shape[1],)  # Depende de la estructura de X_train

        self.min_num_neurons_per_hidden = 10
        self.threshold_num_neurons_per_hidden = 100 #numero de features a partir del cual la búsqueda del número se hace logarítmica

        self.min_num_hidden_layers = 2
        self.max_num_hidden_layers = math.ceil(math.sqrt(self.X_train.shape[1])) #sqroot(nº features)

        self.min_lr = 1e-5
        self.max_lr = 1e-2

        self.bayesian_opt_tuner = None
        self.best_hyperparameters = None
        self.metrics = []
        self.model = None


    def autotune(self):

        ####PRIMERA VUELTA####
        print("Entrada vuelta 1")
        #Se deciden numero de capas ocultas y una primera aprox de lr
        self.bayesian_opt_tuner = kt.BayesianOptimization(
            self.select_num_hidden_layers_and_lr, objective=self.objective, max_trials=self.max_trials, overwrite=self.overwrite,
            directory=self.directory, project_name='num_hidden_layers_and_lr'
        )

        #deja en los atributos de la clase los resultados del fine tuning
        self.search()

        #asignamos resultados de la primera vuelta:
        self.assign_num_hidden_layers_to_model()
        self.assign_lr_to_model()

        ####SEGUNDA VUELTA###
        print("Entrada vuelta 2")
        #Se deciden numero de neuronas por capa y nueva aprox de lr

        if self.X_train.shape[1] >= 10:

            self.bayesian_opt_tuner = kt.BayesianOptimization(
                self.select_num_neurons_per_hidden, objective=self.objective, max_trials=self.max_trials, overwrite=self.overwrite,
                directory=self.directory, project_name='num_neurons_per_hidden'
            )

            # deja en los atributos de la clase los resultados del fine tuning
            self.search()

            # asignamos resultados de la segunda vuelta:
            self.assign_num_neurons_per_hidden_to_model()
            #self.assign_lr_to_model()   Hay que ver como hacerlo
        else:
            self.num_neurons_per_hidden = 10  # SI EL NUMERO DE FEATURES ES INFERIOR A 10, COGER 10 NEURONAS POR CAPA.


        ####TERCERA VUELTA####
        print("Entrada vuelta 3")
        self.bayesian_opt_tuner = kt.BayesianOptimization(
            self.select_optimizer_and_lr, objective=self.objective, max_trials=self.max_trials, overwrite=self.overwrite,
            directory=self.directory, project_name='optimizer_and_lr'
        )

        # deja en los atributos de la clase los resultados del fine tuning
        self.search()

        # asignamos resultados de la primera vuelta:
        self.assign_optimizer_to_model()

        #al fin, se construye el modelo final
        self.model = self.create_and_compile_definitive_model()

    def search(self):
            self.bayesian_opt_tuner.search(self.X_train, self.y_train, epochs=self.num_epochs_tuner,validation_data=self.validation_data)
            self.best_hyperparameters = self.bayesian_opt_tuner.get_best_hyperparameters(num_trials=1)[0].values

    def assign_num_hidden_layers_to_model(self):
        self.num_hidden_layers = self.best_hyperparameters['num_hidden']

    def assign_lr_to_model(self):
        self.lr = self.best_hyperparameters['lr']

    def assign_optimizer_to_model(self):
        self.optimizer = self.best_hyperparameters['optimizer']

    def assign_num_neurons_per_hidden_to_model(self):
        self.num_neurons_per_hidden = self.best_hyperparameters['num_neurons_per_hidden']

    def create_and_compile_definitive_model(self):

        model = Sequential()
        model.add(Input(self.num_neurons_input_layer))

        # Se añaden el resto de capas del modelo
        for i in range(self.num_hidden_layers):
            model.add(Dense(self.num_neurons_per_hidden, activation=self.hidden_activation_function))

        # Se añade capa de salida. La función de activación corresponde al último
        model.add(Dense(self.num_neurons_output_layer, activation=self.output_activation_function))

        #Se vuelve a crear instancia de optimizador, ya que el anterior ya está modificado y no puede ser usado
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)

        # Compiling the model. Hace falta especificar la métrica accuracy para que el objeto history del model.fit contenga tal métrica
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    def create_and_compile_model(self):
        model = Sequential()
        model.add(Input(self.num_neurons_input_layer))

        # Se añaden el resto de capas del modelo
        for i in range(self.num_hidden_layers):
            model.add(Dense(self.num_neurons_per_hidden, activation=self.hidden_activation_function))

        # Se añade capa de salida. La función de activación corresponde al último
        model.add(Dense(self.num_neurons_output_layer, activation=self.output_activation_function))

        # Compiling the model. Hace falta especificar la métrica accuracy para que el objeto history del model.fit contenga tal métrica
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    #Se decide optimizador
    def select_optimizer_and_lr(self,hp):

        optimizer_choice = hp.Choice("optimizer",self.optimizers_list)
        #también se reentrena lr, ya que salia aviso de que si se exploraban mas valores (menores de 1e-5) podia ir mejor
        lr = hp.Float("lr",min_value= (self.lr / 100), max_value= self.lr)

        ###CUIDADO ---> SI LA LISTA CONTIENE ALGUNO QUE NO SEA ESTOS, SALTARÁ ERROR
        if optimizer_choice == 'adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        if optimizer_choice == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        if optimizer_choice == "adamax":
            self.optimizer = keras.optimizers.Adamax(learning_rate=lr)

        model = self.create_and_compile_model()
        return model

    #se deciden numero num_neuronas_por_capa y lr otra vez
    def select_num_neurons_per_hidden(self,hp):
        if self.X_train.shape[1] < self.min_num_neurons_per_hidden:
            self.num_neurons_per_hidden = self.min_num_neurons_per_hidden  # SI EL NUMERO DE FEATURES ES INFERIOR A 10, COGER 10 NEURONAS POR CAPA.
        else:
            if self.X_train.shape[1] > self.threshold_num_neurons_per_hidden:
                self.num_neurons_per_hidden = hp.Int("num_neurons_per_hidden", min_value=self.min_num_neurons_per_hidden,max_value=self.X_train.shape[1],sample='log')  # Si hay muchas features, se hace sample log para que coja valores que representen la gran variación de los posibles valores.
            else:
                self.num_neurons_per_hidden = hp.Int("num_neurons_per_hidden", min_value=self.min_num_neurons_per_hidden,max_value=self.X_train.shape[1])
        model = self.create_and_compile_model()
        return model


    #se deciden num_capas y primera aprox de lr
    def select_num_hidden_layers_and_lr(self,hp):

        self.num_hidden_layers = hp.Int("num_hidden", min_value=self.min_num_hidden_layers, max_value=self.max_num_hidden_layers)

        self.lr = hp.Float("lr", min_value=self.min_lr, max_value=self.max_lr, sampling='log')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr) #No se inicializa en el constructor, ya que nos hace falta primero el valor de lr

        model = self.create_and_compile_model()
        return model

    def train(self):

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=self.validation_split,
            # Se usa validation split para que automáticamente divida el train set. Con validation data hay que separarlo manualmente.
            shuffle=self.shuffle,
            epochs= self.num_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=self.callbacks
        )
        #Obtenemos las metricas
        self.set_metrics(self.callbacks[2])

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