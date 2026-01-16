import tensorflow as tf
# import keras

model = tf.keras.models.load_model('models/actor_22')
# model = keras.models.load_model('models/actor_23_2')
model.save("models/actor_22.keras")

