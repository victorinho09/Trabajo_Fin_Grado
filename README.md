## Librería de AutoML sobre Keras para la creación automática de modelos enfocados en problemas de clasificación


⸻

🚀 Overview

Una librería de AutoML basada en Keras diseñada para generar automáticamente modelos de redes neuronales optimizados para problemas de clasificación.
	•	Combina técnicas heurísticas secuenciales para el ajuste de hiperparámetros.
	•	Interfaz sencilla para usuarios principiantes y opciones de rango avanzadas para expertos.

🎯 Features
	•	Autotune: Busca sucesivamente número de capas, número de neuronas por capa, tasa de aprendizaje, funciones de activación, optimizadores y sus parámetros internos.
	•	Validación interna: validation_split configurable o partición manual usando conjuntos de validación.
	•	Configuración flexible: Permite rangos personalizados de búsqueda de hiperparámetros.


💡 Usage

### Importa la librería
```
from AutoMLClassifier import AutoMLClassifier
```
### Carga tus datos (en este caso, se han cargado datos desde Kaggle)
```
ruta_fichero_kaggle = "uciml/mushroom-classification"
ruta_fichero_csv_interno = "mushrooms.csv"
nombre_fichero_dataset = "mushroom"
nombre_clase_objetivo = "class"

dataset_dir = kagglehub.dataset_download(ruta_fichero_kaggle)
dataset_path = Path(dataset_dir)
data = pd.read_csv(dataset_path / ruta_fichero_csv_interno)
```
### Inicialización de la clase de autotune
```
model_search = AutoMLClassifier(data,"class","mushroom")
```
### Ajuste automático
```
model_search.autotune()
```
### Entrenamiento final del modelo obtenido
```
model_search.train()
```
### Evaluación del modelo
```
loss,precision=model_search.evaluate()
```
### Obtención del modelo
```
final_model = model_search.get_final_model()
```