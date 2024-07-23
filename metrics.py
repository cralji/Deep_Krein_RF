from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import Loss
from tensorflow import cast,reduce_sum,float32
from tensorflow.keras.backend import epsilon

class DiceCoefficient(Metric):
    def __init__(self, target_class=None, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.target_class = target_class
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Asegurarse de que las predicciones son binarias (en caso de que y_pred no esté en formato one-hot)
        # y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        
        if self.target_class is not None:
            # Filtrar por clase específica
            y_true = cast(y_true[..., self.target_class], float32)
            y_pred = cast(y_pred[..., self.target_class], float32)
        else:
            # Convertir one-hot a indices para simplificar cálculo total
            # y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
            y_true = cast(y_true, float32)
            y_pred = cast(y_pred, float32)

        # Calcular la intersección y la unión
        intersection = reduce_sum(y_true * y_pred)
        union = reduce_sum(y_true) + reduce_sum(y_pred)

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        # Calcular el coeficiente de Dice
        dice = (2 * self.intersection + epsilon()) / (self.union + epsilon())
        return dice

    def reset_state(self):
        self.intersection.assign(0)
        self.union.assign(0)


