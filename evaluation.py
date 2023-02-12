import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_loss(history, axis = None):
    """
    Parameters
    ----------
    
        history : 'tf.keras.callbacks.History' object
        axis : 'matplotlib.pyplot.axis' object
    """
    if axis is not None:
        axis.plot(history.epoch, history.history["loss"],
                  label = "Train loss", color = "#191970")
        axis.plot(history.epoch, history.history["val_loss"],
                  label = "Val loss", color = "#00CC33")
        axis.set_title("Loss")
        axis.legend()
    else:
        plt.plot(history.epoch, history.history["loss"],
                 label = "Train loss", color = "#191970")
        plt.plot(history.epoch, history.history["val_loss"],
                 label = "Val loss", color = "#00CC33")
        plt.title("Loss")
        plt.legend()

    
def plot_accuracy(history, axis = None):
    """
    Parameters
    ----------
    
        history : 'tf.keras.callbacks.History' object
        axis : 'matplotlib.pyplot.axis' object
    """
    if axis is not None:
        axis.plot(history.epoch, history.history["accuracy"],
                  label = "Train accuracy", color = "#191970")
        axis.plot(history.epoch, history.history["val_accuracy"],
                  label = "Val accuracy", color = "#00CC33")
        axis.set_ylim(0, 1.1)
        axis.set_title("Accuracy")
        axis.legend()
    else:
        plt.plot(history.epoch, history.history["accuracy"],
                 label = "Train accuracy", color = "#191970")
        plt.plot(history.epoch, history.history["val_accuracy"],
                 label = "Val accuracy", color = "#00CC33")
        plt.title("Accuracy")
        plt.ylim(0, 1.1)
        plt.legend()
    
    
def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Parameters
    ----------
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    
    Returns
    -------
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(layer,
                                                                          batch_size = batch_size)
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum([tf.keras.backend.count_params(p)
                           for p in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(p)
                               for p in model.non_trainable_weights])

    total_memory = (batch_size * shapes_mem_count + internal_model_mem_count\
                    + trainable_count + non_trainable_count)
    
    return total_memory