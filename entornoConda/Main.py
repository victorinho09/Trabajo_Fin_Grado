from pathlib import Path

import kagglehub
import pandas as pd
from Model import Model

ruta_fichero_kaggle = "uciml/mushroom-classification"
ruta_fichero_csv_interno = "mushrooms.csv"
nombre_fichero_dataset = "mushroom"
nombre_clase_objetivo = "class"

dataset_dir = kagglehub.dataset_download(ruta_fichero_kaggle)
dataset_path = Path(dataset_dir)
data = pd.read_csv(dataset_path / ruta_fichero_csv_interno)

model = Model(data,"class","mushroom","directorio_logs",user_max_trials=3)
model.autotune()
model.train()
loss,precision=model.evaluate()
final_model = model.get_final_model()
