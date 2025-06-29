from pathlib import Path

import kagglehub
import pandas as pd
from AutoMLClassifier import AutoMLClassifier

ruta_fichero_kaggle = "uciml/mushroom-classification"
ruta_fichero_csv_interno = "mushrooms.csv"
nombre_fichero_dataset = "mushroom"
nombre_clase_objetivo = "class"

dataset_dir = kagglehub.dataset_download(ruta_fichero_kaggle)
dataset_path = Path(dataset_dir)
data = pd.read_csv(dataset_path / ruta_fichero_csv_interno)

model_search = AutoMLClassifier(data,"class","mushroom")
model_search.autotune()
model_search.train()
loss,precision=model_search.evaluate()
final_model = model_search.get_final_model()
