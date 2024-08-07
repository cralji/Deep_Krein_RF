
from tensorflow.keras.callbacks import Callback

class MetricsCallback(Callback):
    """
    Custom Keras callback to log metrics at the end of training epochs.

    This callback logs metrics such as loss and accuracy at the end of each epoch
    during training. It stores the metrics in a global list `global_metrics` upon
    completion of the final epoch.

    Parameters:
        epochs (int): Total number of epochs for training.

    Methods:
        on_epoch_end(epoch, logs=None):
            Called at the end of each epoch. Logs metrics and stores them in
            `global_metrics` upon completion of the final epoch.
    """

    def __init__(self, epochs):
        """
        Initializes the MetricsCallback instance.

        Parameters:
            epochs (int): Total number of epochs for training.
        """
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function called at the end of each epoch.

        Logs metrics such as loss and accuracy at the end of each epoch.
        Stores the metrics in `global_metrics` upon completion of the final epoch.

        Parameters:
            epoch (int): Current epoch number (0-indexed).
            logs (dict): Dictionary containing the metrics to log.
                Typically contains keys like 'loss' and 'accuracy'.
        """
        if epoch == self.epochs - 1:
            print(f"Final Epoch {epoch + 1}:")
            metrics_dict = {'epoch': epoch + 1}
            for key, value in logs.items():
                print(f"{key}: {value:.4f}")
                metrics_dict[key] = value
            global_metrics.append(metrics_dict)


