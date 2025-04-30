import keras_tuner as kt
import math

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from EpochCumulativeLogger import EpochCumulativeLogger
from GlobalBatchLogger import GlobalBatchLogger
from funciones_auxiliares import get_num_epochs_train, dividir_array

class  Model():
    def __init__(self,X_train,y_train,log_dir,batch_size,num_batches,X_val=None,y_val=None):

        global_batch_logger = GlobalBatchLogger(log_dir)
        global_epoch_logger = EpochCumulativeLogger(log_dir)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

        #Se cogen los datos de validación por paramétro si existen, si no se crean del train set
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

        #atributo del metodo search del tuner
        self.num_epochs_tuner = 5

        self.num_batches = num_batches
        self.num_epochs, self.num_batches_per_epoch = get_num_epochs_train(self.batch_size, self.X_train, self.num_batches,self.validation_split)
        self.num_epochs_tuner = self.num_epochs

        self.history = None #No obtendrá valor hasta que se entrene el modelo
        self.num_hidden_layers = None
        self.lr = None
        self.num_neurons_per_hidden = 10
        self.optimizers_list = ['adam', 'nadam', 'rmsprop','sgd']
        self.optimizer = None
        self.optimizer_name = 'adam'
        self.optimizer_beta1= None
        self.optimizer_beta2 = None
        self.optimizer_epsilon = None
        self.optimizer_rho = None
        self.optimizer_nesterov = None
        self.optimizer_momentum= None
        self.loss = 'categorical_crossentropy' # uso categorical_crossentropy cuando las etiquetas están codificadas con one-hot encoder. Si no usaría: sparse_categ_cross
        self.hidden_activation_function = tf.keras.activations.relu
        self.hidden_activation_function_list = ['relu','leaky_relu','elu','silu']# silu = swish. Misma funcion de activacion para todas las capas. NO merece la pena tener 1 distinta por cada capa. Se incrementa demasiado número de hiperparaemetros que tunear
        self.output_activation_function = 'softmax'
        self.num_neurons_output_layer = self.y_train.shape[1]  # Depende de la estructura de y_train
        self.num_neurons_input_layer = (self.X_train.shape[1],)  # Depende de la estructura de X_train

        self.min_num_neurons_per_hidden = 30
        self.threshold_num_neurons_per_hidden = 100 #numero de features a partir del cual la búsqueda del número se hace logarítmica

        self.min_num_hidden_layers = 2
        self.max_num_hidden_layers = max(4, math.ceil(math.sqrt(self.X_train.shape[1]))) #sqroot(nº features)

        self.min_lr = 1e-6 #Minimiza ConvergenceWarnings en iris almenos
        self.max_lr = 1e-2

        # atributos de parametros pasados al tuner
        self.max_trials = 15
        self.max_trials_activation_function_tuner = len(self.hidden_activation_function_list)
        self.objective = "val_accuracy"
        self.overwrite = True
        self.directory = "bayesian_tuner"

        self.bayesian_opt_tuner = None
        self.best_hyperparameters = None
        self.metrics = []
        self.model = None


    def autotune(self):

        ####PRIMERA VUELTA####
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
        #Se deciden numero de neuronas por capa y nueva aprox de lr

        if self.X_train.shape[1] >= self.min_num_neurons_per_hidden:

            self.bayesian_opt_tuner = kt.BayesianOptimization(
                self.select_num_neurons_per_hidden, objective=self.objective, max_trials=self.max_trials, overwrite=self.overwrite,
                directory=self.directory, project_name='num_neurons_per_hidden'
            )

            # deja en los atributos de la clase los resultados del fine tuning
            self.search()

            # asignamos resultados de la segunda vuelta:
            self.assign_num_neurons_per_hidden_to_model()
        else:
            self.num_neurons_per_hidden = 10  # SI EL NUMERO DE FEATURES ES INFERIOR A 10, COGER 10 NEURONAS POR CAPA.


        ####TERCERA VUELTA
        # Se decide funcion de activacion de las capas ocultas

        self.bayesian_opt_tuner = kt.BayesianOptimization(
            self.select_activation_function, objective=self.objective, max_trials=self.max_trials_activation_function_tuner,overwrite=self.overwrite,
            directory=self.directory, project_name='activation_function'
        )

        # deja en los atributos de la clase los resultados del fine tuning
        self.search()

        # asignamos resultados de la tercera vuelta:
        self.assign_hidden_activation_function_to_model()


        ####Cuarta VUELTA####
        self.bayesian_opt_tuner = kt.BayesianOptimization(
            self.select_optimizer_and_lr, objective=self.objective, max_trials=self.max_trials, overwrite=self.overwrite,
            directory=self.directory, project_name='optimizer_and_lr'
        )

        # deja en los atributos de la clase los resultados del fine tuning
        self.search()


        # asignamos resultados:
        self.assign_lr_to_model()
        self.assign_optimizer_to_model()
        """
    #####Quinta VUELTA####
        self.bayesian_opt_tuner = kt.BayesianOptimization(
            self.select_optimizer_params, objective=self.objective, max_trials=self.max_trials,overwrite=self.overwrite,
            directory=self.directory, project_name='optimizer_params'
        )

        # deja en los atributos de la clase los resultados del fine tuning
        self.search()

        # asignamos resultados:
        self.assign_optimizer_with_params_to_model()
        """
        #al fin, se construye el modelo final
        self.model = self.create_and_compile_definitive_model()

    def search(self):
        #self.bayesian_opt_tuner.oracle.gpr.kernel.set_params(length_scale_bounds=(1e-10, 1e5))
        self.bayesian_opt_tuner.search(self.X_train, self.y_train, epochs=self.num_epochs_tuner,validation_data=self.validation_data,verbose=1)
        self.best_hyperparameters = self.bayesian_opt_tuner.get_best_hyperparameters(num_trials=1)[0].values

    def assign_num_hidden_layers_to_model(self):
        self.num_hidden_layers = self.best_hyperparameters['num_hidden']
        print(f"Número de hidden layers óptimo: {self.best_hyperparameters['num_hidden']}")

    def assign_hidden_activation_function_to_model(self):
        if self.best_hyperparameters['hidden_activation_function'] == 'relu':
            self.hidden_activation_function = tf.keras.activations.relu
        if self.best_hyperparameters['hidden_activation_function'] == "leaky_relu":
            self.hidden_activation_function = tf.keras.activations.leaky_relu
        if self.best_hyperparameters['hidden_activation_function'] == "elu":
            self.hidden_activation_function = tf.keras.activations.elu
        if self.best_hyperparameters['hidden_activation_function'] == "silu":
            self.hidden_activation_function = tf.keras.activations.silu

        print(f"Función de activación óptima: {self.best_hyperparameters['hidden_activation_function']}")

    def assign_lr_to_model(self):
        self.lr = self.best_hyperparameters['lr']
        print(f"Tasa de aprendizaje óptima: {self.best_hyperparameters['lr']}")

    def assign_optimizer_to_model(self):
        if self.best_hyperparameters['optimizer'] == 'adam':
            self.optimizer_name = 'adam'
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        if self.best_hyperparameters['optimizer'] == "rmsprop":
            self.optimizer_name = 'rmsprop'
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        if self.best_hyperparameters['optimizer'] == "nadam":
            self.optimizer_name = 'nadam'
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.lr)
        if self.best_hyperparameters['optimizer'] == "sgd":
            self.optimizer_name = 'sgd'
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr)

        print(f"Optimizador óptimo: {self.best_hyperparameters['optimizer']}")

    def assign_optimizer_with_params_to_model(self):
        if self.optimizer_name == 'adam':
            self.optimizer_beta1 = self.best_hyperparameters['beta1']
            self.optimizer_beta2 = self.best_hyperparameters['beta2']
            self.optimizer_epsilon = self.best_hyperparameters['epsilon']
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr,
                                                      beta_1=self.optimizer_beta1,
                                                      beta_2=self.optimizer_beta2,
                                                      epsilon=self.optimizer_epsilon)
            print(f"----Parámetros del optimizador óptimos:")
            print(f"Beta1: {self.best_hyperparameters['beta1']}")
            print(f"Beta2: {self.best_hyperparameters['beta2']}")
            print(f"Epsilon: {self.best_hyperparameters['epsilon']}")

        if self.optimizer_name == 'rmsprop':
            self.optimizer_rho = self.best_hyperparameters['rho']
            self.optimizer_momentum = self.best_hyperparameters['momentum']
            self.optimizer_epsilon = self.best_hyperparameters['epsilon']

            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr,
                                                         rho=self.optimizer_rho,
                                                         momentum=self.optimizer_momentum,
                                                         epsilon=self.optimizer_epsilon)
            print(f"----Parámetros óptimos:")
            print(f"Epsilon: {self.best_hyperparameters['epsilon']}")
            print(f"Rho: {self.best_hyperparameters['rho']}")
            print(f"Momentum: {self.best_hyperparameters['momentum']}")

        if self.optimizer_name == 'nadam':
            self.optimizer_beta1 = self.best_hyperparameters['beta1']
            self.optimizer_beta2 = self.best_hyperparameters['beta2']
            self.optimizer_epsilon = self.best_hyperparameters['epsilon']
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.lr,
                                                       beta_1=self.optimizer_beta1,
                                                       beta_2=self.optimizer_beta2,
                                                       epsilon=self.optimizer_epsilon)
            print(f"----Parámetros óptimos:")
            print(f"Beta1: {self.best_hyperparameters['beta1']}")
            print(f"Beta2: {self.best_hyperparameters['beta2']}")
            print(f"Epsilon: {self.best_hyperparameters['epsilon']}")

        if self.optimizer_name == 'sgd':
            self.optimizer_momentum = self.best_hyperparameters['momentum']
            self.optimizer_nesterov = self.best_hyperparameters['nesterov']

            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr,
                                                     momentum=self.optimizer_momentum,
                                                     nesterov=self.optimizer_nesterov)

            print(f"----Parámetros óptimos:")
            print(f"Nesterov: {self.best_hyperparameters['nesterov']}")
            print(f"Momentum: {self.best_hyperparameters['momentum']}")

    def assign_num_neurons_per_hidden_to_model(self):
        self.num_neurons_per_hidden = self.best_hyperparameters['num_neurons_per_hidden']
        print(f"Número de neuronas por capa oculta óptimo: {self.best_hyperparameters['num_neurons_per_hidden']}")

    def create_and_compile_definitive_model(self):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.num_neurons_input_layer))

        # Se añaden el resto de capas del modelo
        for i in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(self.num_neurons_per_hidden, activation=self.hidden_activation_function))

        # Se añade capa de salida. La función de activación corresponde al último
        model.add(tf.keras.layers.Dense(self.num_neurons_output_layer, activation=self.output_activation_function))

        # Compiling the model. Hace falta especificar la métrica accuracy para que el objeto history del model.fit contenga tal métrica
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model


    def create_and_compile_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(self.num_neurons_input_layer))

        # Se añaden el resto de capas del modelo
        for i in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(self.num_neurons_per_hidden, activation=self.hidden_activation_function))

        # Se añade capa de salida. La función de activación corresponde al último
        model.add(tf.keras.layers.Dense(self.num_neurons_output_layer, activation=self.output_activation_function))

        # Compiling the model. Hace falta especificar la métrica accuracy para que el objeto history del model.fit contenga tal métrica
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    def select_activation_function(self,hp):
        hidden_activation_function_choice = hp.Choice("hidden_activation_function", self.hidden_activation_function_list)

        ##nuevo optimizer. Será un Adam, ya que la eleccion de optimizador se hace en la siguiente vuelta
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)

        if hidden_activation_function_choice == 'relu':
            self.hidden_activation_function = tf.keras.activations.relu
        if hidden_activation_function_choice == "leaky_relu":
            self.hidden_activation_function = tf.keras.activations.leaky_relu
        if hidden_activation_function_choice == "elu":
            self.hidden_activation_function = tf.keras.activations.elu
        if hidden_activation_function_choice == "silu":
            self.hidden_activation_function = tf.keras.activations.silu

        model = self.create_and_compile_model()
        return model

    #Se decide optimizador
    def select_optimizer_and_lr(self,hp):

        optimizer_choice = hp.Choice("optimizer",self.optimizers_list)
        #también se reentrena lr, ya que salia aviso de que si se exploraban mas valores (menores de 1e-5) podia ir mejor
        lr = hp.Float("lr",min_value= (self.lr / 100), max_value= self.lr*100, sampling='log')

        ###CUIDADO ---> SI LA LISTA CONTIENE ALGUNO QUE NO SEA ESTOS, SALTARÁ ERROR
        if optimizer_choice == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        if optimizer_choice == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        if optimizer_choice == "nadam":
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        if optimizer_choice == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        model = self.create_and_compile_model()
        return model

    #se deciden numero num_neuronas_por_capa y lr otra vez
    def select_num_neurons_per_hidden(self,hp):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        if self.X_train.shape[1] <= self.min_num_neurons_per_hidden:
            self.num_neurons_per_hidden = self.min_num_neurons_per_hidden  # SI EL NUMERO DE FEATURES ES INFERIOR A 10, COGER 10 NEURONAS POR CAPA.
        else:
            if self.X_train.shape[1] > self.threshold_num_neurons_per_hidden:
                self.num_neurons_per_hidden = hp.Int("num_neurons_per_hidden", min_value=self.min_num_neurons_per_hidden,max_value=self.X_train.shape[1],sampling='log')  # Si hay muchas features, se hace sample log para que coja valores que representen la gran variación de los posibles valores.
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

    def select_optimizer_params(self,hp):

        epsilon_choice = np.float32(hp.Float("epsilon",min_value=1e-9, max_value=1e-4, sampling='log'))
        nesterov_choice = np.float32(hp.Boolean("nesterov", default=True))
        beta1_choice = np.float32(hp.Float("beta1",min_value=0.7, max_value=0.99))
        beta2_choice = np.float32(hp.Float("beta2",min_value=0.85, max_value=0.9999))

        if self.optimizer_name == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr,beta_1=beta1_choice,beta_2=beta2_choice,epsilon=epsilon_choice)

        if self.optimizer_name == 'rmsprop':
            rho_choice= hp.Float("rho",min_value=0.7,max_value=0.95)
            momentum_choice= hp.Float("momentum",min_value=0.0,max_value=0.9)
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr,rho=rho_choice,momentum=momentum_choice,epsilon=epsilon_choice)

        if self.optimizer_name == 'nadam':
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.lr,beta_1=beta1_choice,beta_2=beta2_choice,epsilon=epsilon_choice)

        if self.optimizer_name == 'sgd':
            momentum_choice = hp.Float("momentum",min_value=0.7, max_value=0.95)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr,momentum=momentum_choice,nesterov=nesterov_choice)

        model = self.create_and_compile_model()
        return model

    def train(self):

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=self.validation_split,
            # Se usa validation split para que automáticamente divida el train set. Con validation data hay que separarlo manualmente.
            shuffle=self.shuffle,
            epochs= 50,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=self.callbacks
        )
        print("Numero de epocas patra el entrenamiento final:", self.num_epochs)
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

        #for epoch in range(self.num_epochs):
            #info_epoch = [metric_accuracy_per_batch[epoch], metric_loss_per_batch[epoch], metric_val_accuracy[epoch],
            #              metric_val_loss[epoch]]
            #self.metrics.append(info_epoch)
            #print(info_epoch)
            # print(epoch)


    def evaluate(self,X_test_scaled, y_test_encoded):
        self.model.evaluate(X_test_scaled, y_test_encoded)