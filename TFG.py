#Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from keras import Sequential, Input
from keras.src.callbacks import Callback
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo




data_iris = fetch_ucirepo(id=53).data #clasificacion
data_heart_disease = fetch_ucirepo(id=45).data #clasificacion
data_adult = fetch_ucirepo(id=2).data #clasificacion
data_breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17).data #clasificacion
data_bank_marketing = fetch_ucirepo(id=222).data #clasificacion
data_diabetes = fetch_ucirepo(id=296).data #clasificacion
data_mushroom = fetch_ucirepo(id=73).data #clasificacion
data_default_payment = fetch_ucirepo(id=350).data #clasificacion
data_car_evaluation = fetch_ucirepo(id=19).data #clasificacion
data_dry_bean = fetch_ucirepo(id=602).data #clasificacion
data_magic_gamma_telescope = fetch_ucirepo(id=159).data #clasificacion
data_spambase = fetch_ucirepo(id=94).data #clasificacion
data_census_income = fetch_ucirepo(id=20).data #clasificacion
#data_phiusiil_phishing_url_website = fetch_ucirepo(id=967).data #clasificacion. --> NO CABE EN RAM, DA ERROR

#data_bike_sharing = fetch_ucirepo(id=275).data #regresion --> Este dataset funciona muy mal. ¿Mejorará con otra arquitectura?
#data_real_estate_valuation = fetch_ucirepo(id=477).data #regresion --> Problema: Demasiado numero de clases objetivo, y muy pocas instancias, por lo que no da suficiente para que stratify funcione en testset, no consigue meter instancias en train y test set de igual manera
#data_communities_and_crime = fetch_ucirepo(id=183).data #regresion --> Funciona muy mal tmb, muchas clases objetivo

#data_parkinsons_telemonitoring = fetch_ucirepo(id=189).data #regresion -->No coge datos, no contiene nada


def get_y_target_col(data_obj):
    """
    Dado data_obj = fetch_ucirepo(id=...).data, localiza la columna
    de 'targets' y la renombra a 'target'.
    Maneja varios casos:
     - Ya se llama 'target'
     - Se llama 'class'
     - Solo hay una columna en data.targets (la renombra a 'target')
    Devuelve un DataFrame con una sola columna llamada 'target'.
    """
    df_targets = data_obj.targets
    cols = list(df_targets.columns)

    # CASO 1: Si ya existe 'target'
    if 'target' in cols:
        y = df_targets[['target']].copy()
        print("Numero cols en y:", y.columns)
        return y

    # CASO 2: Si existe 'class'
    if 'class' in cols:
        y = df_targets[['class']].rename(columns={'class': 'target'})
        print("Numero cols en y:", y.columns)
        return y

    # CASO 3: Si solo hay 1 columna, la renombramos
    if len(cols) == 1:
        old_col = cols[0]
        y = df_targets.rename(columns={old_col: 'target'})
        return y[['target']]

def preprocessData(data):

    X = pd.DataFrame(data.features, columns=data.feature_names)
    y = get_y_target_col(data)

    #Elimino las filas que contienen que pertenecen a una clase con menos de 2 instancias en todo el dataset,
    #porque sino stratify falla, al no poder balancear la aparición de clases en ambos sets al splittear
    counts = y['target'].value_counts()
    rare_classes = counts[counts < 2].index
    mask = ~y['target'].isin(rare_classes)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    #Filtrar filas donde y['target'] sea NaN
    #(esto crea una máscara True/False, y solo conservas las True)
    mask_not_nan = ~y['target'].isna()
    X = X[mask_not_nan].reset_index(drop=True)
    y = y[mask_not_nan].reset_index(drop=True)

    #Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y #Hace falta stratify porque hay algún dataset (bike_sharing)
        #que contiene muchas clases objetivo y provoca que no esté balanceado el train y el test set.
    )

    #print(y_train.info())
    print("y_train count values:", y_train['target'].nunique())
    print("y_train values:", y_train['target'].unique())
    #print(y_train.head(30))

    '''
    print("X_train shape antes de imputer:", X_train.shape)
    print("¿NaNs en X_train?", X_train.isna().sum())
    '''

    # 4) One-Hot Encoding para y
    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train['target'].values.reshape(-1, 1))
    y_test_encoded  = encoder.transform(y_test['target'].values.reshape(-1, 1))
    #print("y_test_encoded: ", y_test_encoded[1])
    #print("size_y_test_encoded ", len(y_train_encoded[1]))

    # 5) Separar columnas numéricas y categóricas
    numeric_cols_train = X_train.select_dtypes(include=np.number).columns
    categorical_cols_train = X_train.select_dtypes(exclude=np.number).columns

    if len(numeric_cols_train) == 0 and len(categorical_cols_train) == 0:
      raise ValueError("No hay columnas numéricas ni categóricas en X_train")

    numeric_cols_test = X_test.select_dtypes(include=np.number).columns
    categorical_cols_test = X_test.select_dtypes(exclude=np.number).columns

    if len(numeric_cols_test) == 0 and len(categorical_cols_test) == 0:
      raise ValueError("No hay columnas numéricas ni categóricas en X_test")

    if(len(categorical_cols_train) == 0):
        print("NO hay columnas categoricas en X_train")

    # Imputar numéricas
    imputer_num = SimpleImputer(strategy='mean')
    if len(numeric_cols_train) > 0:
      X_train_num_imputed = imputer_num.fit_transform(X_train[numeric_cols_train])
      X_test_num_imputed  = imputer_num.transform(X_test[numeric_cols_test])

      # Escalar numéricas
      scaler = StandardScaler()
      X_train_num_scaled = scaler.fit_transform(X_train_num_imputed)
      X_test_num_scaled  = scaler.transform(X_test_num_imputed)

    else:
      X_train_num_scaled = np.zeros((X_train.shape[0], 0))
      X_test_num_scaled  = np.zeros((X_test.shape[0], 0))


    # Imputar categóricas
    imputer_cat = SimpleImputer(strategy='most_frequent')
    if len(categorical_cols_train) > 0:
      X_train_cat_imputed = imputer_cat.fit_transform(X_train[categorical_cols_train])
      X_test_cat_imputed  = imputer_cat.transform(X_test[categorical_cols_test])
    else:
      X_train_cat_imputed = np.zeros((X_train.shape[0], 0))
      X_test_cat_imputed  = np.zeros((X_test.shape[0], 0))

    # Codificar categóricas

    if(len(categorical_cols_train) != 0):
        print("Hay columnas categoricas en X_train")
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_cat_encoded = ohe.fit_transform(X_train_cat_imputed)
        X_test_cat_encoded  = ohe.transform(X_test_cat_imputed)

    else:
        X_train_cat_encoded = np.zeros((X_train.shape[0], 0))
        X_test_cat_encoded  = np.zeros((X_test.shape[0], 0))

    # Concatenar numéricas escaladas + categóricas codificadas
    X_train_final = np.concatenate([X_train_num_scaled, X_train_cat_encoded], axis=1)
    X_test_final  = np.concatenate([X_test_num_scaled, X_test_cat_encoded],  axis=1)

    # Revisar si hay NaN en la salida final
    print("¿Hay NaN en X_train_final?", np.isnan(X_train_final).any())

    return X_train_final, X_test_final, y_train_encoded, y_test_encoded

#numero de neuronas rango (capa inicial) : raiz del numero de features - numero de features
#numero capas:


# Creation of the ANN of example to try out the batch metrics in TensorBoard
def create_model(X_train, y_train):
    print("y_train_shape: ", y_train.shape[1])
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden Layer 1
        Dense(32, activation='relu'),  # Hidden Layer 2
        Dense(y_train.shape[1], activation='softmax')  # Output Layer with softmax for multiclass clasification
    ])

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
        'accuracy'])  # uso categorical_crossentropy cuando las etiquetas están codificadas con one-hot encoder. Si no usaría: sparse_categ_cross

    return model


# Con esta callback, registramos la perdida y precisión acumulada tras cada época.
class EpochCumulativeLogger(Callback):
    def __init__(self, log_dir):
        super(EpochCumulativeLogger, self).__init__()
        self.log_dir = log_dir

        # Variables para acumulados (se actualizarán época a época)
        self.cumulative_loss = 0.0
        self.cumulative_accuracy = 0.0
        self.epoch_count = 0

    def on_train_begin(self, logs=None):
        # Creamos un writer para logs de TensorBoard
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        """
        Se llama al terminar cada época.
        'logs' contiene la métrica 'loss' y 'accuracy' (y val_loss, val_accuracy si procede).
        """
        if logs is not None:
            # Obtenemos la pérdida y la accuracy promedio de la época
            epoch_loss = logs.get('loss', 0.0)
            epoch_acc = logs.get('accuracy', 0.0)

            # Actualizamos la suma acumulada
            self.cumulative_loss += epoch_loss
            self.cumulative_accuracy += epoch_acc
            self.epoch_count += 1

            # Calculamos la media acumulada en todas las épocas hasta ahora
            avg_cumulative_loss = self.cumulative_loss / self.epoch_count
            avg_cumulative_accuracy = self.cumulative_accuracy / self.epoch_count

            # Registramos en TensorBoard
            tf.summary.scalar('epoch_cumulative_loss', avg_cumulative_loss, step=self.epoch_count)
            tf.summary.scalar('epoch_cumulative_accuracy', avg_cumulative_accuracy, step=self.epoch_count)

            # Forzamos la escritura
            self.writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()


'''
class GlobalBatchLogger(Callback):
    def __init__(self, log_dir):
        super(GlobalBatchLogger, self).__init__()
        self.log_dir = log_dir
        self.global_step = 0

        # Variables para acumulados
        self.cumulative_loss = 0.0
        self.cumulative_accuracy = 0.0
        self.total_samples = 0  # para saber cuántas muestras se han procesado

    def on_train_begin(self, logs=None):
        # Creamos el escritor de resúmenes de TensorFlow al inicio del entrenamiento
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.writer.set_as_default()

    def on_batch_end(self, batch, logs=None):
        """
        Se llama al terminar cada batch.
        ‘batch’ es el índice local del batch en la época, pero usamos
        self.global_step para numerar de forma continua a lo largo
        de todo el entrenamiento.
        """
        if logs is not None:
            # Obtenemos las métricas *promediadas* que Keras ha calculado hasta este batch
            batch_loss = logs.get('loss', 0.0)
            batch_acc = logs.get('accuracy', 0.0)

            # Determinamos cuántas muestras se incluyeron en este batch
            batch_size = logs.get('size', None)
            if batch_size is None:
                # Si no aparece en logs, tomamos la configuración de la clase
                batch_size = self.params.get('batch_size', 1)

            # Actualizamos la suma acumulada de (loss * nº muestras) y (acc * nº muestras)
            self.cumulative_loss += batch_loss * batch_size
            self.cumulative_accuracy += batch_acc * batch_size
            self.total_samples += batch_size

            # Calculamos la media acumulada (hasta este batch)
            avg_cumulative_loss = self.cumulative_loss / self.total_samples
            avg_cumulative_accuracy = self.cumulative_accuracy / self.total_samples

            # Registramos exclusivamente la métrica acumulada
            tf.summary.scalar('cumulative_loss', data=avg_cumulative_loss, step=self.global_step)
            tf.summary.scalar('cumulative_accuracy', data=avg_cumulative_accuracy, step=self.global_step)

        # Forzamos la escritura en los ficheros de logs
        self.writer.flush()

        # Incrementamos el contador global de batches
        self.global_step += 1

    def on_train_end(self, logs=None):
        # Cerramos el writer al finalizar
        self.writer.close()
'''


# Esta función te da el numero de epocas que se ejecutaran si quieres que se ejecuten un numero de batches concreto.
# No es un numero de epocas exacto, ya que redondeamos hacia arriba. 3,1 epochs -> 4 epochs
def getNumEpochsTrain(batch_size, X_train_scaled, desired_batches):
    total_samples = X_train_scaled.shape[0]
    # si 1 epoca es una vuelta entera a todos los samples. Y cada batch ejecuta batch_size instancias
    numBatchesPorEpoch = math.ceil(total_samples / batch_size)
    if (numBatchesPorEpoch > desired_batches):
        print(
            "No se llega a ejecutar 1 epoca entera --> numEpochs = 0????")  # preguntar: ¿Queremos comparar ejecuciones con las epochs?
    numEpochs = math.ceil(desired_batches / numBatchesPorEpoch)
    return numEpochs


'''
def compute_steps_for_batches(
    desired_batches,
    X_train_scaled,
    batch_size=16
):
    """
    Dado un número deseado de batches (desired_batches), calcula
    el número de steps que se pueden entrenar, sin pasarse
    del total de batches en X_train_scaled.
    - desired_batches: cuántos batches queremos.
    - X_train_scaled: datos de entrenamiento.
    - batch_size: tamaño de lote.

    Retorna steps_for_n_batches, que es el mínimo entre desired_batches y
    la cantidad real de batches que hay.
    """
    total_samples = X_train_scaled.shape[0]
    total_batches = math.ceil(total_samples / batch_size)

    # Para no pasarnos de la época, usamos el mínimo
    steps_for_n_batches = min(desired_batches, total_batches)

    # También puedes forzar que sea al menos 1
    if steps_for_n_batches < 1:
        steps_for_n_batches = 1

    return steps_for_n_batches

'''


def train_and_evaluate(dataset, num_batches, log_dir, batch_size):
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = preprocessData(dataset)
    model = create_model(X_train_scaled, y_train_encoded)

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )

    # global_batch_logger = GlobalBatchLogger(log_dir)
    global_epoch_logger = EpochCumulativeLogger(log_dir)

    # steps_per_epoch = compute_steps_for_batches(num_batches,X_train_scaled,batch_size)
    numEpochs = getNumEpochsTrain(batch_size, X_train_scaled, num_batches)

    history = model.fit(
        X_train_scaled,
        y_train_encoded,
        validation_data=(X_test_scaled, y_test_encoded),
        epochs=numEpochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[tb_callback, global_epoch_logger]
    )
    loss, accuracy = model.evaluate(X_test_scaled, y_test_encoded)
    print(f"Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")
    return history

train_and_evaluate(data_iris,500, "logs/fit/iris/500batches",16)
train_and_evaluate(data_heart_disease,500, "logs/fit/heart_disease/500batches",16)
train_and_evaluate(data_breast_cancer_wisconsin_diagnostic,500, "logs/fit/breast_cancer_wisconsin_diagnostic/500batches",16)
train_and_evaluate(data_bank_marketing,500, "logs/fit/bank_marketing/500batches",16)
train_and_evaluate(data_diabetes,500, "logs/fit/diabetes/500batches",16)
train_and_evaluate(data_adult,500, "logs/fit/adult/500batches",16)
train_and_evaluate(data_mushroom,500, "logs/fit/mushroom/500batches",16)
train_and_evaluate(data_default_payment,500, "logs/fit/default_payment/500batches",16)
#train_and_evaluate(data_bike_sharing,500, "logs/fit/bike_sharing/500batches",16)
train_and_evaluate(data_dry_bean,500, "logs/fit/dry_bean/500batches",16)
train_and_evaluate(data_spambase,500, "logs/fit/spambase/500batches",16)
train_and_evaluate(data_magic_gamma_telescope,500, "logs/fit/magic_gamma_telescope/500batches",16)
train_and_evaluate(data_car_evaluation,500, "logs/fit/car_evaluation/500batches",16)
train_and_evaluate(data_census_income,500, "logs/fit/census_income/500batches",16)

# %load_ext tensorboard
# %tensorboard --logdir logs/fit