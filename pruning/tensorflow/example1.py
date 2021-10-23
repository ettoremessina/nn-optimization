import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow_model_optimization.sparsity.keras import ConstantSparsity
from tensorflow_model_optimization.sparsity.keras import PolynomialDecay

from support.trim_insignificant_weights import *
from support.scatter_graph import *

def build_mlp_regression_model():
    inputs = keras.Input(shape=(1,))
    hidden = inputs
    hidden = layers.Dense(32, use_bias=True, activation='relu')(hidden)
    hidden = layers.Dense(64, use_bias=True, activation='relu')(hidden)
    hidden = layers.Dense(32, use_bias=True, activation='relu')(hidden)
    outputs = layers.Dense(1, use_bias=True)(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp_regression_model")
    return model

fx_gen_ds = lambda x: x**2 #generating function of the dataset
x_dataset = np.arange(-2., 2, 0.005)
y_dataset = fx_gen_ds(x_dataset)

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

model_org = build_mlp_regression_model()
model_org.summary()

batch_size=80
epochs=100
loss=losses.MeanSquaredError(reduction='auto', name='mean_squared_error')
optimizer = optimizers.Adam()

model_org.compile(loss=loss, optimizer=optimizer)

model_org.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=0)

y_pred_org = model_org.predict(x_test)[:, 0]
error_org = loss(y_test, y_pred_org).numpy()

num_of_w_org, num_of_nz_w_org, num_of_z_w_org = \
    inspect_weigths('original (unpruned)', model_org)
unzipped_size_org, zipped_size_org = retrieve_size_of_model(model_org)

attempt_infos = []

ai_org = AttemptInfo (
    'original (unpruned)',
    num_of_w_org, num_of_nz_w_org, num_of_z_w_org,
    unzipped_size_org, zipped_size_org,
    y_pred_org, error_org)
ai_org.print()
attempt_infos.append(ai_org)

end_step = np.ceil(len(x_train) / batch_size).astype(np.int32) * epochs

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
    model_pruning.compile(loss=loss, optimizer=optimizer)
    callbacks_pruning = retrieve_callbacks_for_pruning()
    model_pruning.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks_pruning,
        verbose=0)
    model_pruned = extract_pruned_model(model_pruning)
    num_of_w_pruned, num_of_nz_w_pruned, num_of_z_w_pruned = \
        inspect_weigths(ac.name, model_pruned)
    y_pred_pruned = model_pruned.predict(x_test)[:, 0]
    error_pruned = loss(y_test, y_pred_pruned).numpy()
    unzipped_size_pruned, zipped_size_pruned = retrieve_size_of_model(model_pruned)

    mi_pruned = AttemptInfo (
        ac.name,
        num_of_w_pruned, num_of_nz_w_pruned, num_of_z_w_pruned,
        unzipped_size_pruned, zipped_size_pruned,
        y_pred_pruned, error_pruned)
    mi_pruned.print()
    attempt_infos.append(mi_pruned)

print('')
print('*** Final recap ***')
print_attempt_infos(attempt_infos)
scatter_attempt_infos(attempt_infos, x_test, y_test)
