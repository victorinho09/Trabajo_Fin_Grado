import math
import os




def pintar_resultado_en_fichero(datos_entrenamiento):
    directorio_resultados_pruebas_internas = "./graficas_tfg"
    os.makedirs(directorio_resultados_pruebas_internas, exist_ok=True)
    ruta = f"{datos_entrenamiento['nombre_dataset']}_{datos_entrenamiento['user_num_epochs_tuner']}_{datos_entrenamiento['max_trials']}"
    ruta_fichero = os.path.join(directorio_resultados_pruebas_internas,ruta )

    if os.path.exists(ruta_fichero):
        with open(ruta_fichero, "a") as f:
            for i in range(len(datos_entrenamiento)):
                print(f"{list(datos_entrenamiento.keys())[i]} = {datos_entrenamiento[list(datos_entrenamiento.keys())[i]]}", file=f)
            print("####################################", file=f)
    else:
        # Con w, se sobreescribe el fichero entero
        with open(ruta_fichero, "w") as f:
            for i in range(len(datos_entrenamiento)):
                print(f"{list(datos_entrenamiento.keys())[i]} = {datos_entrenamiento[list(datos_entrenamiento.keys())[i]]}", file=f)
            print("####################################", file=f)

def dividir_array(arr, n):
    """
    Divide la lista 'arr' en sublistas de 'n' elementos.
    Si la longitud de 'arr' no es múltiplo de n, la última sublista contendrá
    los elementos restantes.
    """
    return [arr[i:i+n] for i in range(0, len(arr), n)]

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
        return y

    # CASO 2: Si existe 'class'
    if 'class' in cols:
        y = df_targets[['class']].rename(columns={'class': 'target'})
        return y

    # CASO 3: Si solo hay 1 columna, la renombramos
    if len(cols) == 1:
        old_col = cols[0]
        y = df_targets.rename(columns={old_col: 'target'})
        return y[['target']]

# Esta función te da el numero de epocas que se ejecutaran si quieres que se ejecuten un numero de batches concreto.
# No es un numero de epocas exacto, ya que redondeamos hacia arriba. 3,1 epochs -> 4 epochs
# def get_num_epochs_train(
#         batch_size,
#         X_train_scaled,
#         validation_split_value
# ):
#     num_batches_per_epoch = get_num_batches_per_epoch(validation_split_value=validation_split_value,filas=X_train_scaled.shape[0],batch_size=batch_size)
#     numEpochs = math.ceil(desired_batches / num_batches_per_epoch)
#     return numEpochs, num_batches_per_epoch

def get_num_batches_per_epoch(validation_split_value,filas,batch_size):
    total_samples =  ((1-validation_split_value) * filas)
    # si 1 epoca es una vuelta entera a todos los samples. Y cada batch ejecuta batch_size instancias
    return math.ceil(total_samples / batch_size)

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