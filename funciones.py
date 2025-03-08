import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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