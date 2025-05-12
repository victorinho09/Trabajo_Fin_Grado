import tensorflow as tf

# Con esta callback, registramos la perdida y precisión acumulada tras cada época.
class EpochCumulativeLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(EpochCumulativeLogger, self).__init__()
        self.log_dir = log_dir

        # Variables para acumulados (se actualizarán época a época)
        self.cumulative_loss = 0.0
        self.cumulative_accuracy = 0.0
        self.epoch_count = 0

    def on_train_begin(self, logs=None):
        self.epoch_count = 0
        self.cumulative_loss = 0.0
        self.cumulative_accuracy = 0.0
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