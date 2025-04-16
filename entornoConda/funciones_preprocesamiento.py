import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from funciones_auxiliares import get_y_target_col


def preprocess_data(data,nombre_fichero_info_dataset):

    directorio_info_datasets = "./info_datasets"
    os.makedirs(directorio_info_datasets,exist_ok=True)
    ruta_fichero = os.path.join(directorio_info_datasets, nombre_fichero_info_dataset)

    X = pd.DataFrame(data.features, columns=data.feature_names)
    y = get_y_target_col(data)


    #Elimino las filas que contienen que pertenecen a una clase con menos de 2 instancias en el dataset
    #porque sino stratify falla, al no poder balancear la aparición de clases en ambos sets al splittear
    counts = y['target'].value_counts()
    rare_classes = counts[counts < 2].index
    mask = ~y['target'].isin(rare_classes)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    #Filtrar filas donde y['target'] sea NaN
    #(esto crea una máscara True/False, y solo conservas las True)
    mask_not_nan = ~y['target'].isna()
    X_not_nan = X[mask_not_nan].reset_index(drop=True)
    y_not_nan = y[mask_not_nan].reset_index(drop=True)

    #Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_not_nan, y_not_nan, test_size=0.2, random_state=42, stratify=y #Hace falta stratify porque hay algún dataset (bike_sharing)
        #que contiene muchas clases objetivo y provoca que no esté balanceado el train y el test set.
    )

    #Con w, se sobreescribe el fichero entero
    with open(ruta_fichero, "w") as f:
        f.write(f"y_train_count values: {y_train['target'].nunique()}\n")

    with open(ruta_fichero, "a") as f:
        f.write(f"y_train values: {y_train['target'].unique()}\n")
        y_train.info(buf=f)

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
        mensaje_info = "No hay columnas numéricas ni categóricas en X_train"
        with open(ruta_fichero, "a") as f:
            f.write(f"{mensaje_info} \n")
        raise ValueError(mensaje_info)


    numeric_cols_test = X_test.select_dtypes(include=np.number).columns
    categorical_cols_test = X_test.select_dtypes(exclude=np.number).columns

    if len(numeric_cols_test) == 0 and len(categorical_cols_test) == 0:
        mensaje_info = "No hay columnas numéricas ni categóricas en X_test"
        with open(ruta_fichero, "a") as f:
            f.write(f"{mensaje_info} \n")
        raise ValueError(mensaje_info)

    if(len(categorical_cols_train) == 0):
        mensaje_info = "No hay columnas categoricas en X_train"
        with open(ruta_fichero, "a") as f:
            f.write(f"{mensaje_info} \n")

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
        mensaje_info = "Hay columnas categoricas en X_train"
        with open(ruta_fichero, "a") as f:
            f.write(f"{mensaje_info} \n")
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
    with open(ruta_fichero, "a") as f:
        f.write(f"¿Hay NaN en X_train_final? {np.isnan(X_train_final).any()} \n")

    with open(ruta_fichero, "a") as f:
        for i, col in enumerate(X.columns):
            f.write(f"{i}: {col} \n")

    return X_train_final, X_test_final, y_train_encoded, y_test_encoded