import argparse
import tensorflow as tf
import os
project_path = os.path.dirname(os.path.realpath("../../"))
import sys
sys.path.append(project_path)
from src.method.wavelets.multiresolution_cnn_architecture import MultiresolutionCNN
from src.method.wavelets.tfreader import make_dataset
from src.method.utils import load_parameters
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiresolution CNN model training')
    parser.add_argument("model",
                        type=str,
                        help="Model's name")
    parser.add_argument("tr_tfrecord",
                        type=str,
                        help="Training TFRecord file")
    parser.add_argument("val_tfrecord",
                        type=str,
                        help="Validation TFrecord file")
    parser.add_argument("parameters",
                        type=str,
                        help="JSON file containing the parameters of the model")
    parser.add_argument("--test_tfrecord",
                        type=str,
                        help="Testing TFRecord file",
                        default=None)
    args = parser.parse_args()

    # Use only the GPU specified in the parameters
    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.debugging.set_log_device_placement(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


    # Load parameters of the model
    parameters = load_parameters(args.parameters)
    if "gpu" in parameters.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] = parameters["gpu"]

    model = MultiresolutionCNN(parameters)
    # tf.keras.utils.plot_model(model, 'shallow_cnn_mlw_classification.png', show_shapes=True)

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=parameters['learning_rate'])


    def train_loop(features, labels, training=False):
        # Define the GradientTape context
        with tf.GradientTape() as tape:
            # Get the probabilities
            predictions = model(features, training)
            # labels = tf.dtypes.cast(labels, tf.float32)
            # Calculate the loss
            loss = loss_func(labels, predictions)
        # Get the gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Update the weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, predictions


    # Training loop
    # 1/ Iterate each epoch. An epoch is one pass through the dataset
    # 2/ Whithin an epoch, iterate over each example in the training Dataset.
    # 3/ Calculate model's loss and gradients
    # 4/ Use an optimizer to update the model's variables
    # 5/ Keep track of stats and repeat

    train_loss_results = []
    train_accuracy_results = []

    validation_loss_results = []
    validation_accuracy_results = []

    # checkpoint_path = "models/ShallowCNN/model_ep_{}.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    num_epochs = parameters['epochs']

    initial_loss = 10.0
    for epoch in range(num_epochs):
        print("Current epoch: {}".format(epoch))
        checkpoint_path = "models/{}/{}/model_001.ckpt".format(args.model, parameters["chunk_size"])
        # checkpoint_dir = os.path.dirname(checkpoint_path)

        d_train = make_dataset(args.tr_tfrecord,
                               parameters['buffer_size'],
                               parameters['batch_size'],
                               1)
        d_val = make_dataset(args.val_tfrecord,
                             1024,
                             1,
                             1)

        # Training metrics
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        # Validation metrics
        val_epoch_loss_avg = tf.keras.metrics.Mean()
        val_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        tr_step = 0

        # Training loop
        for step, (x, y) in enumerate(d_train):
            loss, y_ = train_loop(x, y, True)

            # Track progress
            epoch_loss_avg(loss)
            epoch_accuracy(y, y_)
            print("Iteration step: {}; Loss: {:.3f}, Accuracy: {:.3%}".format(tr_step,
                                                                              epoch_loss_avg.result(),
                                                                              epoch_accuracy.result()))
            tr_step += 1

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in d_val:
            val_logits = model(x_batch_val, False)
            val_loss = loss_func(y_batch_val, val_logits)

            # Update metrics
            val_epoch_loss_avg(val_loss)
            val_epoch_accuracy(y_batch_val, val_logits)

        val_acc = val_epoch_accuracy.result()
        val_loss = val_epoch_loss_avg.result()
        print('Epoch: {}; Validation loss {}; acc: {}'.format(epoch, val_loss, val_acc))

        validation_loss_results.append(val_loss)
        validation_accuracy_results.append(val_acc)

        if float(val_loss) < initial_loss:
            initial_loss = float(val_loss)
            model.save_weights(checkpoint_path)  # Save only the weights

    model.load_weights(checkpoint_path)
    test_epoch_loss_avg = tf.keras.metrics.Mean()
    test_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    y_actual_test = []
    y_pred_test = []
    # Evaluate model on the test set
    if args.test_tfrecord is not None:
        d_test = make_dataset(args.test_tfrecord,
                              1,
                              1,
                              1)

        for x_batch_test, y_batch_test in d_test:
            test_logits = model(x_batch_test, False)
            test_loss = loss_func(y_batch_test, test_logits)

            # For the confusion matrix
            y_pred = tf.argmax(test_logits, axis=-1)
            y_pred_test.extend(y_pred)
            y_actual_test.extend(y_batch_test)

            # Update metrics
            test_epoch_loss_avg(test_loss)
            test_epoch_accuracy(y_batch_test, test_logits)

        test_acc = test_epoch_accuracy.result()
        test_loss = test_epoch_loss_avg.result()
        print('Test loss {}; acc: {}'.format(test_loss, test_acc))

        cm = confusion_matrix(y_actual_test, y_pred_test)
        print("Confusion Matrix:\n {}".format(cm))