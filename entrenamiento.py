import tensorflow as tf
from EpochCumulativeLogger import EpochCumulativeLogger
from GlobalBatchLogger import GlobalBatchLogger
from funciones import create_model, get_num_epochs_train, dividir_array
from funciones_preprocesamiento import preprocess_data


def train_and_evaluate(dataset, num_batches, log_dir, batch_size):
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = preprocess_data(dataset)
    model = create_model(X_train_scaled, y_train_encoded)

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch'
    )

    global_batch_logger = GlobalBatchLogger(log_dir)
    global_epoch_logger = EpochCumulativeLogger(log_dir)

    # steps_per_epoch = compute_steps_for_batches(num_batches,X_train_scaled,batch_size)

    validation_split_value = 0.2
    #Hay que tener en cuenta que el validation split es 0.2, por lo que realmente se usa solo el 80% del dataset de entrenamiento
    num_epochs, num_batches_per_epoch = get_num_epochs_train(batch_size, X_train_scaled, num_batches, validation_split_value)

    history = model.fit(
        X_train_scaled,
        y_train_encoded,
        validation_split=validation_split_value, #Se usa validation split para que automáticamente divida el train set. Con validation data hay que separarlo manualmente.
        shuffle=True, #Para que baraje los datos antes de la división del val set
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[tb_callback, global_epoch_logger, global_batch_logger]
    )

    # metricas_conjunto_validacion = history.history.keys()
    # metric_accuracy = history.history["accuracy"]
    # metric_loss = history.history["loss"]
    metric_val_accuracy = history.history["val_accuracy"]
    metric_val_loss = history.history["val_loss"]
    metric_loss_per_batch = dividir_array(global_batch_logger.batch_loss_acum,num_batches_per_epoch)
    metric_accuracy_per_batch = dividir_array(global_batch_logger.batch_accuracy_acum,num_batches_per_epoch)

    #print("Número de elementos en val_accuracy: ", len(metric_val_accuracy))
    #print("Número de elementos en val_loss: ", len(metric_val_loss))
    #print("Número de elementos/listas en loss_per_batch: ", len(metric_loss_per_batch))
    #print("Número de elementos/listas en accuracy_per_batch: ", len(metric_accuracy_per_batch))




    #print("Número de batches por época: ",num_batches_per_epoch)
    #print("metric_loss_per_batch: ", metric_loss_per_batch)
    #print("metric_accuracy_per_batch",metric_accuracy_per_batch)

    metricas = []

    for epoch in range(num_epochs):
        info_epoch = [ metric_accuracy_per_batch[epoch], metric_loss_per_batch[epoch], metric_val_accuracy[epoch], metric_val_loss[epoch] ]
        metricas.append(info_epoch)
        #print(info_epoch)
        #print(epoch)



    #print("History: ",metricas_conjunto_validacion)
    #print("Val_accuracy: ", metric_val_accuracy)
    #print("Val_loss: ", metric_val_loss)

    loss, accuracy = model.evaluate(X_test_scaled, y_test_encoded)
    return metricas