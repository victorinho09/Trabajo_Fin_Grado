import math
import time

import keras_tuner as kt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from EpochCumulativeLogger import EpochCumulativeLogger
from GlobalBatchLogger import GlobalBatchLogger
from funciones_auxiliares import get_num_batches_per_epoch

class  Model():
    def __init__(self,X_train,y_train,log_dir,X_val=None,y_val=None,
                 user_batch_size=None, #
                 user_num_epochs=None, #
                 user_max_trials= None, #
                 user_min_num_hidden_layers: int = None,    #
                 user_max_num_hidden_layers: int = None,    #
                 user_min_num_neurons_per_hidden: int = None,
                 user_max_num_neurons_per_hidden: int = None,
                 user_optimizers_list: list = None, #
                 user_hidden_activation_function_list: list = None, #
                 user_lr = None #
                 ):

        self.validation_split = 0.3  # Se usa validation split para que automáticamente divida el train set. Con validation data hay que separarlo manualmente.
        self.shuffle = True  # Para que baraje los datos antes de la división del val set
        # Se cogen los datos de validación por paramétro si existen, si no se crean del train set
        if (X_val is None) or (y_val is None):
            # hacer split del train
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train,
                                                                                  test_size=self.validation_split)  # Se genera el conjunto de validacion y se ponen como argumentos todos los datasets necesarios
        else:
            self.X_val = X_val
            self.y_val = y_val
            self.X_train = X_train
            self.y_train = y_train
        self.validation_data = (self.X_val, self.y_val)

        # global_batch_logger = GlobalBatchLogger(log_dir + "/batch/" + time.strftime("%Y%m%d-%H%M%S"))
        # global_epoch_logger = EpochCumulativeLogger(log_dir + "/epoch/" + time.strftime("%Y%m%d-%H%M%S"))
        # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir + "/tensorboard/" + time.strftime("%Y%m%d-%H%M%S"), histogram_freq=1, update_freq='epoch')

        # atributos de parametros pasados al fit method de funcion train
        if user_batch_size is not None:
            self.batch_size = user_batch_size
        else:
            self.batch_size = 16
        self.verbose = 1
        self.log_dir = log_dir

        #Se hace solo para el entrenamiento final del modelo
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode="min",
            restore_best_weights=True,
            min_delta=1e-4
        )
        #tb_callback, global_epoch_logger, global_batch_logger,
        self.callbacks = [early_stop]


        self.num_clases_target = self.y_train.shape[1]
        #Se entrena siempre 1000 epocas ya que se aplica early stopping
        self.num_epochs_final_training = 1000

        # self.num_batches = user_num_batches

        if self.num_clases_target == 2:
            self.loss = "binary_crossentropy"
        else:
            self.loss = 'categorical_crossentropy' # uso categorical_crossentropy cuando las etiquetas están codificadas con one-hot encoder. Si no usaría: sparse_categ_cross

        ################HASTA AQUI HIPERPARAMETROS DE ENTRENAMIENTO

        self.history = None #No obtendrá valor hasta que se entrene el modelo

        if user_lr is not None:
            self.lr = user_lr
            self.search_lr = False
        else:
            self.lr = 1e-5
            self.search_lr = True

        self.min_lr = 1e-6  # Minimiza ConvergenceWarnings en iris almenos
        self.max_lr = 1e-2

        self.initialize_hidden_function_variables(user_hidden_activation_function_list)

        self.num_neurons_output_layer = self.y_train.shape[1]  # Depende de la estructura de y_train
        self.num_neurons_input_layer = (self.X_train.shape[1],)  # Depende de la estructura de X_train
        if self.num_neurons_output_layer == 2:
            self.output_activation_function = "sigmoid"
        else:
            self.output_activation_function = 'softmax'

        self.initialize_num_neurons_per_hidden_variables(user_min_num_neurons_per_hidden,user_max_num_neurons_per_hidden)

        self.num_hidden_layers = None
        self.initialize_num_hidden_layers_variables(user_min_num_hidden_layers,user_max_num_hidden_layers)

        self.initialize_optimizer_variables(user_optimizers_list)

        self.initialize_tuner_variables(user_max_trials,user_num_epochs)

        self.best_hyperparameters = None
        self.metrics = []
        self.model = None

    def initialize_num_neurons_per_hidden_variables(self,user_min_num_neurons_per_hidden=None,user_max_num_neurons_per_hidden=None):
        self.num_neurons_per_hidden = 10
        self.search_num_neurons_per_hidden = True
        self.threshold_num_neurons_per_hidden_less_than = 10
        self.min_num_neurons_per_hidden = 5
        self.threshold_num_neurons_per_hidden_log = 100  # numero de features a partir del cual la búsqueda del número se hace logarítmica
        self.use_user_param_values = False
        self.max_num_neurons_per_hidden = math.ceil((2/3) * self.X_train.shape[1] + self.y_train.shape[1])

        if user_min_num_neurons_per_hidden is not None:
             if user_max_num_neurons_per_hidden is not None:
                 #Se ha recibido min y max neurons per hidden layers por parametro
                 if user_min_num_neurons_per_hidden < user_max_num_neurons_per_hidden:
                     self.min_num_neurons_per_hidden = user_min_num_neurons_per_hidden
                     self.max_num_neurons_per_hidden = user_max_num_neurons_per_hidden
                 elif user_min_num_neurons_per_hidden == user_max_num_neurons_per_hidden:
                     self.num_neurons_per_hidden = user_max_num_neurons_per_hidden
                     self.search_num_neurons_per_hidden = False
                 else:
                     print(f"El máximo introducido ({user_max_num_neurons_per_hidden}) es menor que el mínimo introducido ({user_min_num_neurons_per_hidden}).Se cogerán valores por defecto.")
             else:
                 #solo se ha recibido min neurons per hidden layers por parametro
                 self.min_num_neurons_per_hidden = user_min_num_neurons_per_hidden
        else:
             if user_max_num_neurons_per_hidden is not None:
                 #Solo se ha recibido max neurons per hidden layers por parametro
                 if user_max_num_neurons_per_hidden > self.min_num_neurons_per_hidden:
                     self.max_num_neurons_per_hidden = user_max_num_neurons_per_hidden
                 else:
                     print(f"El mínimo por defecto ({self.min_num_neurons_per_hidden}) > {user_max_num_neurons_per_hidden}. Se cogerán valores por defecto")
             print(f"Valor de max_num_neurons: {self.max_num_neurons_per_hidden}. Debe devolver none")

    def initialize_optimizer_variables(self,user_optimizers_list = None):

        #Traduccion de los optimizadores disponibles por tf.keras
        self.available_optimizers = {
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "adam": tf.keras.optimizers.Adam,
            "adamw": tf.keras.optimizers.AdamW,
            "adamax": tf.keras.optimizers.Adamax,
            "adafactor": tf.keras.optimizers.Adafactor,
            "nadam": tf.keras.optimizers.Nadam,
            "adadelta": tf.keras.optimizers.Adadelta,
            "adagrad": tf.keras.optimizers.Adagrad,
            "ftrl": tf.keras.optimizers.Ftrl,
            "lion": tf.keras.optimizers.Lion,
        }

        if user_optimizers_list:
            optimizers_list_lowercase = [s.lower() for s in user_optimizers_list]
            self.optimizers_list =  optimizers_list_lowercase
        else:
            self.optimizers_list = ['adam','sgd','rmsprop']

        self.optimizer = None #NO se hace en constructor porque falta lr
        self.optimizer_name = self.optimizers_list[0] #Porque de momento se coge el primero para hacer pruebas
        self.optimizer_beta1= None
        self.optimizer_beta2 = None
        self.optimizer_rho = None
        self.optimizer_nesterov = None
        self.optimizer_momentum= None

    def initialize_hidden_function_variables(self,user_hidden_activation_function_list = None):
        # Traduccion de las funciones de activacion disponibles por tf.keras
        self.available_hidden_activation_functions = {
            "leaky_relu": tf.keras.activations.leaky_relu,
            "mish": tf.keras.activations.mish,
            "relu": tf.keras.activations.relu,
            "sigmoid": tf.keras.activations.sigmoid,
            "softmax": tf.keras.activations.softmax,
            "softplus": tf.keras.activations.softplus,
            "softsign": tf.keras.activations.softsign,
            "tanh": tf.keras.activations.tanh,
            "selu": tf.keras.activations.selu,
            "elu": tf.keras.activations.elu,
            "exponential": tf.keras.activations.exponential,
            "linear": tf.keras.activations.linear,
            "gelu": tf.keras.activations.gelu,
            "silu": tf.keras.activations.swish,
            "swish": tf.keras.activations.swish,
        }

        # Se comprueba si el parametro opcional viene dado o no
        if user_hidden_activation_function_list:
            # Se pasan a minúsculas todas las strings de la lista
            hidden_activation_function_list_lowercase = [s.lower() for s in user_hidden_activation_function_list]
            self.hidden_activation_function_list = hidden_activation_function_list_lowercase
        else:
            self.hidden_activation_function_list = ['relu','elu', 'silu']
        self.hidden_activation_function = self.available_hidden_activation_functions[self.hidden_activation_function_list[0]]

    def initialize_tuner_variables(self,user_max_trials=None,user_num_epochs=None):

        if user_num_epochs is not None:
            self.num_epochs_tuner = user_num_epochs
            #self.num_batches_per_epoch = get_num_batches_per_epoch(validation_split_value=self.validation_split,filas=self.X_train.shape[0],batch_size=self.batch_size)
        else:
            self.num_epochs_tuner = 10
            # self.num_epochs, self.num_batches_per_epoch = get_num_epochs_train(self.batch_size, self.X_train, self.num_batches,self.validation_split)


        # atributos de parametros pasados al tuner
        if user_max_trials is not None:
            self.max_trials = user_max_trials
        else:
            self.max_trials = 15

        self.bayesian_opt_tuner = None

        #El número de max_trials para buscar función de activación será igual a la longitud de la lista de funciones de activación
        self.max_trials_activation_function_tuner = len(self.hidden_activation_function_list)

        ##El número mínimo de max_trials para buscar el optimizador debe ser la longitud de la lista de optimizadores. Pero el máximo podrá ser cualquiera
        if (user_max_trials is not None) and (user_max_trials >= len(self.optimizers_list)):
            self.max_trials_optimizer_tuner = user_max_trials
        else:
            self.max_trials_optimizer_tuner = len(self.optimizers_list)

        self.objective = "val_loss"
        self.overwrite = True
        self.directory = "autotune"

    def initialize_num_hidden_layers_variables(self,user_min_num_hidden_layers=None,user_max_num_hidden_layers=None):
        self.min_num_hidden_layers = 2
        self.max_num_hidden_layers = 6  #Habia demasiadas capas si se ponia la raiz del numero de features
        self.search_num_hidden_layers = True

        if user_min_num_hidden_layers is not None:
            if user_max_num_hidden_layers is not None:
                #Se ha recibido min y max hidden layers por parametro
                if user_min_num_hidden_layers < user_max_num_hidden_layers:
                    self.min_num_hidden_layers = user_min_num_hidden_layers
                    self.max_num_hidden_layers = user_max_num_hidden_layers
                elif user_min_num_hidden_layers == user_max_num_hidden_layers:
                    self.num_hidden_layers = user_max_num_hidden_layers
                    self.search_num_hidden_layers = False
                else:
                    print(f"El valor máximo introducido({user_max_num_hidden_layers}) es menor que el mínimo introducido({user_min_num_hidden_layers}). Se cogerán valores por defecto.")

            else:
                #solo se ha recibido min hidden layers por parametro
                if user_min_num_hidden_layers < self.max_num_hidden_layers:
                    self.min_num_hidden_layers = user_min_num_hidden_layers
                else:
                    print(f"Mínimo de capas ocultas introducido ({user_min_num_hidden_layers}) es mayor que el máximo de capas por defecto ({self.max_num_hidden_layers}). Se cogerán valores por defecto")
        else:
            if user_max_num_hidden_layers is not None:
                #Solo se ha recibido max hidden layers por parametro
                if user_max_num_hidden_layers > self.min_num_hidden_layers:
                    self.max_num_hidden_layers = user_max_num_hidden_layers
                else:
                    print(f"Máximo de capas ocultas introducido ({user_max_num_hidden_layers}) es menor o igual que el mínimo por defecto ({self.min_num_hidden_layers}). Se cogerán valores por defecto")

    def autotune(self):

        ####PRIMERA VUELTA####
        if self.search_num_hidden_layers:
            if self.search_lr:
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
            else:
                self.bayesian_opt_tuner = kt.BayesianOptimization(
                    self.select_num_hidden_layers, objective=self.objective, max_trials=self.max_trials,overwrite=self.overwrite,
                    directory=self.directory, project_name='num_hidden_layers_and_lr'
                )
                # deja en los atributos de la clase los resultados del fine tuning
                self.search()

                # asignamos resultados de la primera vuelta:
                self.assign_num_hidden_layers_to_model()
        else:
            print(f"\nSaltando búsqueda de num_hidden_layers. El usuario ya ha fijado el valor por defecto: {self.num_hidden_layers}\n")
        self.debugHyperparams()

        ####SEGUNDA VUELTA###
        #Se deciden numero de neuronas por capa

        if self.search_num_neurons_per_hidden:
            if self.X_train.shape[1] >= self.threshold_num_neurons_per_hidden_less_than:

                self.bayesian_opt_tuner = kt.BayesianOptimization(
                    self.select_num_neurons_per_hidden, objective=self.objective, max_trials=self.max_trials, overwrite=self.overwrite,
                    directory=self.directory, project_name='num_neurons_per_hidden'
                )

                # deja en los atributos de la clase los resultados del fine tuning
                self.search()

                # asignamos resultados de la segunda vuelta:
                self.assign_num_neurons_per_hidden_to_model()
            else:
                self.num_neurons_per_hidden = self.threshold_num_neurons_per_hidden_less_than  # SI EL NUMERO DE FEATURES ES INFERIOR A 10, COGER 10 NEURONAS POR CAPA.
        else:
            print(f"\nSaltando búsqueda de num_neurons_per_hidden. El usuario ya ha fijado el valor por defecto: {self.num_neurons_per_hidden}\n")

        self.debugHyperparams()

        ####TERCERA VUELTA
        # Se decide funcion de activacion de las capas ocultas
        self.bayesian_opt_tuner = kt.GridSearch(
            self.select_activation_function, objective=self.objective, max_trials=self.max_trials_activation_function_tuner,overwrite=self.overwrite,
            directory=self.directory, project_name='activation_function'
        )

        # deja en los atributos de la clase los resultados del fine tuning
        self.search()

        # asignamos resultados de la tercera vuelta:
        self.assign_hidden_activation_function_to_model()

        self.debugHyperparams()

        ####Cuarta VUELTA####
        if self.search_lr:
            self.bayesian_opt_tuner = kt.BayesianOptimization(
                self.select_optimizer_and_lr, objective=self.objective, max_trials=self.max_trials_optimizer_tuner, overwrite=self.overwrite,
                directory=self.directory, project_name='optimizer_and_lr'
            )

            # deja en los atributos de la clase los resultados del fine tuning
            self.search()

            # asignamos resultados:
            self.assign_lr_to_model()
            self.assign_optimizer_to_model()
        else:
            self.bayesian_opt_tuner = kt.BayesianOptimization(
                self.select_optimizer, objective=self.objective, max_trials=self.max_trials_optimizer_tuner,overwrite=self.overwrite,
                directory=self.directory, project_name='optimizer_and_lr'
            )

            # deja en los atributos de la clase los resultados del fine tuning
            self.search()

            # asignamos resultados:
            self.assign_optimizer_to_model()

        self.debugHyperparams()

        #####Quinta VUELTA####
        #Hacer vuelta extra de elección de lr más en detalle con el optimizador correcto
        if self.search_lr:
            self.bayesian_opt_tuner = kt.BayesianOptimization(
                self.select_lr, objective=self.objective, max_trials=self.max_trials,overwrite=self.overwrite,
                directory=self.directory, project_name='lr'
            )

            # deja en los atributos de la clase los resultados del fine tuning
            self.search()

            # asignamos resultados:
            self.assign_lr_to_model()
        self.debugHyperparams()

        #####SEXTA VUELTA ####
        self.bayesian_opt_tuner = kt.BayesianOptimization(
            self.select_optimizer_params, objective=self.objective, max_trials=self.max_trials,overwrite=self.overwrite,
            directory=self.directory, project_name='optimizer_params'
        )

        # deja en los atributos de la clase los resultados del fine tuning
        self.search()

        # asignamos resultados:
        self.assign_optimizer_with_params_to_model()

        self.debugHyperparams()

        #al fin, se construye el modelo final
        self.model = self.create_and_compile_model()
        self.debugHyperparams()

    def search(self):
        #self.bayesian_opt_tuner.oracle.gpr.kernel.set_params(length_scale_bounds=(1e-10, 1e5))
        self.bayesian_opt_tuner.search(self.X_train, self.y_train, epochs=self.num_epochs_tuner,batch_size=self.batch_size,validation_data=self.validation_data,verbose=self.verbose)
        self.best_hyperparameters = self.bayesian_opt_tuner.get_best_hyperparameters(num_trials=1)[0].values

    def assign_num_hidden_layers_to_model(self):
        self.num_hidden_layers = self.best_hyperparameters['num_hidden']
        print(f"Número de hidden layers óptimo: {self.best_hyperparameters['num_hidden']}")

    def assign_hidden_activation_function_to_model(self):
        # Se traduce la funcion de activacion de string -> funcion de tf.keras
        self.hidden_activation_function = self.available_hidden_activation_functions[self.best_hyperparameters['hidden_activation_function']]
        print(f"Función de activación óptima: {self.best_hyperparameters['hidden_activation_function']}")

    def assign_lr_to_model(self):
        self.lr = float(self.best_hyperparameters['lr'])
        print(f"Tasa de aprendizaje óptima: {self.best_hyperparameters['lr']}")

    def assign_optimizer_to_model(self):
        # Se traduce el optimizador de string -> funcion de tf.keras
        self.optimizer_name = self.best_hyperparameters['optimizer']
        print(f"Optimizador óptimo: {self.best_hyperparameters['optimizer']}")

    def assign_optimizer_with_params_to_model(self):
        if self.optimizer_name == 'adamw':
            self.optimizer_beta1 = self.best_hyperparameters['beta1']
            self.optimizer_beta2 = self.best_hyperparameters['beta2']
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate = self.lr,
                                                      beta_1=self.optimizer_beta1,
                                                      beta_2=self.optimizer_beta2)
            print(f"----Parámetros del optimizador óptimos:")
            print(f"Beta1: {self.best_hyperparameters['beta1']}")
            print(f"Beta2: {self.best_hyperparameters['beta2']}")

        if self.optimizer_name == 'adamax':
            self.optimizer_beta1 = self.best_hyperparameters['beta1']
            self.optimizer_beta2 = self.best_hyperparameters['beta2']
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate = self.lr,
                                                      beta_1=self.optimizer_beta1,
                                                      beta_2=self.optimizer_beta2)
            print(f"----Parámetros del optimizador óptimos:")
            print(f"Beta1: {self.best_hyperparameters['beta1']}")
            print(f"Beta2: {self.best_hyperparameters['beta2']}")

        if self.optimizer_name == 'adafactor':
            self.optimizer = tf.keras.optimizers.Adafactor(learning_rate = self.lr)
            print(f"----Parámetros del optimizador óptimos:")
            print("Ningún parámetro tuneado")

        if self.optimizer_name == 'adadelta':
            self.optimizer_rho = self.best_hyperparameters['rho']
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = self.lr,
                                                      rho= self.optimizer_rho)
            print(f"----Parámetros del optimizador óptimos:")
            print(f"Rho: {self.best_hyperparameters['rho']}")

        if self.optimizer_name == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate = self.lr)
            print(f"----Parámetros del optimizador óptimos:")
            print("Ningún parámetro tuneado")

        if self.optimizer_name == 'ftrl':
            self.optimizer = tf.keras.optimizers.Ftrl(learning_rate = self.lr)
            print(f"----Parámetros del optimizador óptimos:")
            print("Ningún parámetro tuneado")


        if self.optimizer_name == 'lion':
            self.optimizer_beta1 = self.best_hyperparameters['beta1']
            self.optimizer_beta2 = self.best_hyperparameters['beta2']
            self.optimizer = tf.keras.optimizers.Lion(learning_rate = self.lr,
                                                      beta_1=self.optimizer_beta1,
                                                      beta_2=self.optimizer_beta2)
            print(f"----Parámetros del optimizador óptimos:")
            print(f"Beta1: {self.best_hyperparameters['beta1']}")
            print(f"Beta2: {self.best_hyperparameters['beta2']}")

        if self.optimizer_name == 'adam':
            self.optimizer_beta1 = self.best_hyperparameters['beta1']
            self.optimizer_beta2 = self.best_hyperparameters['beta2']
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr,
                                                      beta_1=self.optimizer_beta1,
                                                      beta_2=self.optimizer_beta2)
            print(f"----Parámetros del optimizador óptimos:")
            print(f"Beta1: {self.best_hyperparameters['beta1']}")
            print(f"Beta2: {self.best_hyperparameters['beta2']}")

        if self.optimizer_name == 'rmsprop':
            self.optimizer_rho = self.best_hyperparameters['rho']
            self.optimizer_momentum = self.best_hyperparameters['momentum']
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr,
                                                         rho=self.optimizer_rho,
                                                         momentum=self.optimizer_momentum)
            print(f"----Parámetros óptimos:")
            print(f"Rho: {self.best_hyperparameters['rho']}")
            print(f"Momentum: {self.best_hyperparameters['momentum']}")

        if self.optimizer_name == 'nadam':
            self.optimizer_beta1 = self.best_hyperparameters['beta1']
            self.optimizer_beta2 = self.best_hyperparameters['beta2']
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.lr,
                                                       beta_1=self.optimizer_beta1,
                                                       beta_2=self.optimizer_beta2)
            print(f"----Parámetros óptimos:")
            print(f"Beta1: {self.best_hyperparameters['beta1']}")
            print(f"Beta2: {self.best_hyperparameters['beta2']}")

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

        ##nuevo optimizer. Será un Adam, ya que la eleccion de optimizador se hace en la siguiente vuelta. #Se traduce la funcion de activacion de string -> funcion de tf.keras
        self.optimizer = self.available_optimizers[self.optimizers_list[0]](learning_rate=self.lr)

        #Se traduce la funcion de activacion de string -> funcion de tf.keras
        self.hidden_activation_function = self.available_hidden_activation_functions[hidden_activation_function_choice]

        model = self.create_and_compile_model()
        return model

    #Se decide optimizador
    def select_optimizer_and_lr(self,hp):

        optimizer_choice = hp.Choice("optimizer",self.optimizers_list)
        #también se reentrena lr, ya que salia aviso de que si se exploraban mas valores (menores de 1e-5) podia ir mejor
        lr = hp.Float("lr",min_value= (self.lr / 10), max_value= (self.lr *10), sampling='log')
        if lr >= self.max_lr:
            #Si se pasa del limite, regresar al valor anterior, que está acotado en los límites
            lr = self.lr
        # Se traduce el optimizador de string -> funcion de tf.keras
        self.optimizer = self.available_optimizers[optimizer_choice](learning_rate= lr)

        model = self.create_and_compile_model()
        return model

    def select_optimizer(self,hp):
        optimizer_choice = hp.Choice("optimizer", self.optimizers_list)
        # Se traduce el optimizador de string -> funcion de tf.keras
        self.optimizer = self.available_optimizers[optimizer_choice](learning_rate=self.lr)

        model = self.create_and_compile_model()
        return model

    def select_lr(self,hp):
        lr = hp.Float("lr", min_value=(self.lr / 100), max_value=self.lr, sampling='log')

        if self.optimizer_name == 'adamw':
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)

        if self.optimizer_name == 'adamax':
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate=lr)
    
        if self.optimizer_name == 'adafactor':
            self.optimizer = tf.keras.optimizers.Adafactor(learning_rate=lr)
    
        if self.optimizer_name == 'adadelta':
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
    
        if self.optimizer_name == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    
        if self.optimizer_name == 'ftrl':
            self.optimizer = tf.keras.optimizers.Ftrl(learning_rate=lr)
    
        if self.optimizer_name == 'lion':
            self.optimizer = tf.keras.optimizers.Lion(learning_rate=lr)
    
        if self.optimizer_name == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
        if self.optimizer_name == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    
        if self.optimizer_name == 'nadam':
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    
        if self.optimizer_name == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

        model = self.create_and_compile_model()
        return model

    #se deciden numero num_neuronas_por_capa y lr otra vez
    def select_num_neurons_per_hidden(self,hp):
        # Se traduce el optimizador de string -> funcion de tf.keras
        self.optimizer = self.available_optimizers[self.optimizers_list[0]](learning_rate=self.lr)

        if self.max_num_neurons_per_hidden <= self.min_num_neurons_per_hidden:
            self.num_neurons_per_hidden = self.min_num_neurons_per_hidden
        else:
            if self.max_num_neurons_per_hidden > self.threshold_num_neurons_per_hidden_log:
                self.num_neurons_per_hidden = hp.Int("num_neurons_per_hidden", min_value=self.min_num_neurons_per_hidden,max_value=self.max_num_neurons_per_hidden,sampling='log')  # Si hay muchas features, se hace sample log para que coja valores que representen la gran variación de los posibles valores.

            else:
                self.num_neurons_per_hidden = hp.Int("num_neurons_per_hidden", min_value=self.min_num_neurons_per_hidden,max_value=self.max_num_neurons_per_hidden)

        model = self.create_and_compile_model()
        return model


    #se deciden num_capas y primera aprox de lr
    def select_num_hidden_layers_and_lr(self,hp):

        self.num_hidden_layers = hp.Int("num_hidden", min_value=self.min_num_hidden_layers, max_value=self.max_num_hidden_layers)

        self.lr = hp.Float("lr", min_value=self.min_lr, max_value=self.max_lr, sampling='log')
        # Se traduce el optimizador de string -> funcion de tf.keras
        self.optimizer = self.available_optimizers[self.optimizers_list[0]](learning_rate=self.lr) #No se inicializa en el constructor, ya que nos hace falta primero el valor de lr. Se coge el primer optimizador de la lista

        model = self.create_and_compile_model()
        return model

    def select_num_hidden_layers(self,hp):
        self.num_hidden_layers = hp.Int("num_hidden", min_value=self.min_num_hidden_layers,
                                        max_value=self.max_num_hidden_layers)

        # Se traduce el optimizador de string -> funcion de tf.keras
        self.optimizer = self.available_optimizers[self.optimizers_list[0]](
            learning_rate=self.lr)  # No se inicializa en el constructor, ya que nos hace falta primero el valor de lr. Se coge el primer optimizador de la lista

        model = self.create_and_compile_model()
        return model

    def select_optimizer_params(self,hp):

        if self.optimizer_name == 'adamw':
            beta1_choice = float(hp.Float("beta1", min_value=0.85, max_value=0.99))
            beta2_choice = float(hp.Float("beta2", min_value=0.85, max_value=0.9999))
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate = self.lr,beta_1=beta1_choice,beta_2=beta2_choice,clipnorm=1.0)

        if self.optimizer_name == 'adamax':
            beta1_choice = float(hp.Float("beta1", min_value=0.85, max_value=0.99))
            beta2_choice = float(hp.Float("beta2", min_value=0.85, max_value=0.9999))
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate = self.lr,beta_1=beta1_choice,beta_2=beta2_choice,clipnorm=1.0)

        if self.optimizer_name == 'adafactor':
            self.optimizer = tf.keras.optimizers.Adafactor(learning_rate = self.lr,clipnorm=1.0)

        if self.optimizer_name == 'adadelta':
            rho_choice = float(hp.Float("rho", min_value=0.8, max_value=0.95))
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate = self.lr,rho=rho_choice,clipnorm=1.0)

        if self.optimizer_name == 'adagrad':
            #Merece la pena finetunear algun hiperparametro aqui?
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate = self.lr,clipnorm=1.0)

        if self.optimizer_name == 'ftrl':
            #Merece la pena finetunear algun hiperparametro aqui?
            self.optimizer = tf.keras.optimizers.Ftrl(learning_rate = self.lr,clipnorm=1.0)

        if self.optimizer_name == 'lion':
            beta1_choice = float(hp.Float("beta1", min_value=0.85, max_value=0.99))
            beta2_choice = float(hp.Float("beta2", min_value=0.85, max_value=0.9999))
            self.optimizer = tf.keras.optimizers.Lion(learning_rate = self.lr,beta_1=beta1_choice,beta_2=beta2_choice,clipnorm=1.0)

        if self.optimizer_name == 'adam':
            beta1_choice = float(hp.Float("beta1", min_value=0.85, max_value=0.99))
            beta2_choice = float(hp.Float("beta2", min_value=0.85, max_value=0.9999))
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr,beta_1=beta1_choice,beta_2=beta2_choice,clipnorm=1.0)

        if self.optimizer_name == 'rmsprop':
            rho_choice = float(hp.Float("rho", min_value=0.8, max_value=0.95))
            momentum_choice= float(hp.Float("momentum",min_value=0.7,max_value=0.9))
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr,rho=rho_choice,momentum=momentum_choice,clipnorm=1.0)

        if self.optimizer_name == 'nadam':
            beta1_choice = float(hp.Float("beta1", min_value=0.85, max_value=0.99))
            beta2_choice = float(hp.Float("beta2", min_value=0.85, max_value=0.9999))
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.lr,beta_1=beta1_choice,beta_2=beta2_choice,clipnorm=1.0)

        if self.optimizer_name == 'sgd':
            momentum_choice = float(hp.Float("momentum",min_value=0.7, max_value=0.95))
            nesterov_choice = hp.Boolean("nesterov", default=True)
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr,momentum=momentum_choice,nesterov=nesterov_choice,clipnorm=1.0)

        model = self.create_and_compile_model()
        return model

    def train(self):

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=self.validation_data,
            shuffle=self.shuffle,
            epochs= self.num_epochs_final_training, #Se ponen muchas épocas, ya que se quiere que el modelo se entrene bien, independientemente del resto de tiempo de búsquedda de hiperparámetros. Con el early stopping para de entrenar
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=self.callbacks
        )
        self.num_epochs_trained = len(self.history.history['loss'])
        #Obtenemos las metricas
        # self.set_metrics(self.callbacks[2])


    #def set_metrics(self,global_batch_logger):

    #    metric_val_accuracy = self.history.history["val_accuracy"]
    #    metric_val_loss = self.history.history["val_loss"]
    #    metric_loss_per_batch = dividir_array(global_batch_logger.batch_loss_acum, self.num_batches_per_epoch)
    #    metric_accuracy_per_batch = dividir_array(global_batch_logger.batch_accuracy_acum, self.num_batches_per_epoch)

    def evaluate(self,X_test, y_test):
        #Devuelve una lista, elemento 0 -> loss, elemento 1 -> accuracy
        print("Se hace la evaluacion:")
        #Por pantalla se imprime la loss del ultimo batch ejecutado por evaluate. Pero la funcion devuelve la perdida media de todos los batches
        return self.model.evaluate(X_test, y_test,batch_size=self.batch_size)

    def get_final_hyperparams_and_params(self):
        return [self.lr,self.optimizer_name,self.hidden_activation_function.__name__,self.num_neurons_per_hidden,self.num_hidden_layers,self.num_epochs_trained,self.max_trials]

    def debugHyperparams(self):
        print(f"#########DEBUG##########")
        print(f"Número de capas: {self.num_hidden_layers}")
        print(f"Learning Rate: {self.lr}")
        print(f"Número de neuronas por capa: {self.num_neurons_per_hidden}")
        print(f"Función de activación: {self.hidden_activation_function.__name__}")
        print(f"Optimizador: {self.optimizer_name}")
