from keras.src.callbacks import Callback
import tensorflow as tf

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