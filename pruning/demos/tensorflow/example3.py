import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow_model_optimization.sparsity.keras import ConstantSparsity
from tensorflow_model_optimization.sparsity.keras import PolynomialDecay

from support.trim_insignificant_weights import *
from support.scatter_graph import *

import tensorflow_datasets as tfds

def configure_for_performance(ds, batch_size):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds

def build_cnn_model(img_height, img_width, num_classes):
    inputs = keras.Input(shape=(img_height, img_width, 3))
    hidden = inputs
    hidden = layers.Conv2D(32, 3, activation='tanh')(hidden)
    hidden = layers.MaxPooling2D()(hidden)
    hidden = layers.Conv2D(32, 3, activation='tanh')(hidden)
    hidden = layers.MaxPooling2D()(hidden)
    hidden = layers.Conv2D(32, 3, activation='tanh')(hidden)
    hidden = layers.MaxPooling2D()(hidden)
    hidden = layers.Flatten()(hidden)
    hidden = layers.Dense(128, activation='tanh')(hidden)
    hidden = layers.Dropout(0.5)(hidden)
    outputs = layers.Dense(num_classes)(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_model")
    return model

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

total_number_of_training_images = len(train_ds)
total_number_of_validation_images = len(val_ds)
total_number_of_test_images = len(test_ds)
print('Total number of Training images:', total_number_of_training_images)
print('Total number of Validation images:', total_number_of_validation_images)
print('Total number of Test images:', total_number_of_test_images)

epochs = 15
batch_size = 100
img_height = 180
img_width = 180
num_classes = metadata.features['label'].num_classes

img_size = (img_height, img_width)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, img_size)/255.0, y))
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, img_size)/255.0, y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, img_size)/255.0, y))

train_ds = configure_for_performance(train_ds, batch_size)
val_ds = configure_for_performance(val_ds, batch_size)
test_ds = configure_for_performance(test_ds, total_number_of_test_images)

model_org = build_cnn_model(img_height, img_width, num_classes)
model_org.summary()

loss=losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer=optimizers.Adam()

model_org.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

callbacks = []
model_org.fit(
    train_ds,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=2)

image_batch_test, label_batch_test = next(iter(test_ds))
image_batch_test = image_batch_test.numpy()
label_batch_test = label_batch_test.numpy()

predicted_batch_org = model_org.predict(image_batch_test)
predicted_batch_org = tf.squeeze(predicted_batch_org).numpy()
predicted_ids_org = np.argmax(predicted_batch_org, axis=-1)
predicted_ok_org = np.sum(label_batch_test == predicted_ids_org)
predicted_failed_org = len(label_batch_test) - predicted_ok_org

num_of_w_org, num_of_nz_w_org, num_of_z_w_org = \
    inspect_weigths('original (unpruned)', model_org)
unzippedh5_size_org, zippedh5_size_org = retrieve_size_of_model(model_org)
unzippedlt_size_org, zippedlt_size_org = retrieve_size_of_lite_model(model_org)

attempt_infos = []

ai_org = AttemptInfo (
    'original (unpruned)',
    num_of_w_org, num_of_nz_w_org, num_of_z_w_org,
    unzippedh5_size_org, zippedh5_size_org,
    unzippedlt_size_org, zippedlt_size_org,
    predicted_ids_org, predicted_failed_org / total_number_of_test_images)
ai_org.print()
attempt_infos.append(ai_org)

end_step = np.ceil(total_number_of_training_images / batch_size).astype(np.int32) * epochs

attempt_configs = [
    AttemptConfig('poly decay 10/50', PolynomialDecay(
        initial_sparsity=0.10,
        final_sparsity=0.50,
        begin_step=0,
        end_step=end_step)),
    AttemptConfig('poly decay 20/50', PolynomialDecay(
        initial_sparsity=0.20,
        final_sparsity=0.50,
        begin_step=0,
        end_step=end_step)),
    AttemptConfig('poly decay 30/60', PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=0.60,
        begin_step=0,
        end_step=end_step)),
    AttemptConfig('poly decay 30/70', PolynomialDecay(
        initial_sparsity=0.30,
        final_sparsity=0.70,
        begin_step=0,
        end_step=end_step)),
    AttemptConfig('poly decay 40/50', PolynomialDecay(
        initial_sparsity=0.40,
        final_sparsity=0.50,
        begin_step=0,
        end_step=end_step)),
    AttemptConfig('poly decay 10/90', PolynomialDecay(
        initial_sparsity=0.10,
        final_sparsity=0.90,
        begin_step=0,
        end_step=end_step)),
    AttemptConfig('const sparsity 0.1', ConstantSparsity(
        target_sparsity=0.1, begin_step=0
        )),
    AttemptConfig('const sparsity 0.4', ConstantSparsity(
        target_sparsity=0.4, begin_step=0
        )),
    AttemptConfig('const sparsity 0.5', ConstantSparsity(
        target_sparsity=0.5, begin_step=0
        )),
    AttemptConfig('const sparsity 0.6', ConstantSparsity(
        target_sparsity=0.6, begin_step=0
        )),
    AttemptConfig('const sparsity 0.9', ConstantSparsity(
        target_sparsity=0.9, begin_step=0
        ))
]

for ac in attempt_configs:
    model_pruning = build_pruning_model(model_org, ac.pruning_schedule)
    model_pruning.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    callbacks_pruning = retrieve_callbacks_for_pruning()
    model_pruning.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks_pruning,
        verbose=2)
    model_pruned = extract_pruned_model(model_pruning)
    num_of_w_pruned, num_of_nz_w_pruned, num_of_z_w_pruned = \
        inspect_weigths(ac.name, model_pruned)
    predicted_batch_pruned = model_pruned.predict(image_batch_test)
    predicted_batch_pruned = tf.squeeze(predicted_batch_pruned).numpy()
    predicted_ids_pruned = np.argmax(predicted_batch_pruned, axis=-1)
    predicted_ok_pruned = np.sum(label_batch_test == predicted_ids_pruned)
    predicted_failed_pruned = len(label_batch_test) - predicted_ok_pruned
    unzippedh5_size_pruned, zippedh5_size_pruned = retrieve_size_of_model(model_pruned)
    unzippedlt_size_pruned, zippedlt_size_pruned = retrieve_size_of_lite_model(model_pruned)

    mi_pruned = AttemptInfo (
        ac.name,
        num_of_w_pruned, num_of_nz_w_pruned, num_of_z_w_pruned,
        unzippedh5_size_pruned, zippedh5_size_pruned,
        unzippedlt_size_pruned, zippedlt_size_pruned,
        predicted_ids_pruned, predicted_failed_pruned / total_number_of_test_images)
    mi_pruned.print()
    attempt_infos.append(mi_pruned)

print('')
print('*** Final recap ***')
print_attempt_infos(attempt_infos)
