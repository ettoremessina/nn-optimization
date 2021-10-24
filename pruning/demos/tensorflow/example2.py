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

import pandas as pd

def build_samples(seq, sample_length):
    df = pd.DataFrame(seq)
    cols = list()
    for i in range(sample_length, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, 1):
        cols.append(df.shift(-i))

    aggregate = pd.concat(cols, axis=1)
    aggregate.dropna(inplace=True)

    X_train, y_train = aggregate.values[:, :-1], aggregate.values[:, -1]

    return X_train, y_train

def compute_forecast(model):
    y_forecast = np.array([])
    to_predict_flat = np.array(y_train_timeseries[-sample_length:])
    for i in range(forecast_length):
        to_predict = to_predict_flat.reshape((1, sample_length, 1))
        prediction = model.predict(to_predict, verbose=0)[0]
        y_forecast = np.append(y_forecast, prediction)
        to_predict_flat = np.delete(to_predict_flat, 0)
        to_predict_flat = np.append(to_predict_flat, prediction)
    return y_forecast

def build_lstm_model(sample_length):
    inputs = keras.Input(shape=(sample_length, 1))
    hidden = inputs
    hidden = layers.LSTM(80, use_bias=True, activation='tanh')(hidden)
    outputs = layers.Dense(1, use_bias=True)(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs, name="long_short_term_memory_model")
    return model

ft_gen_ts = lambda t: 2.0 * np.sin(t/10.0) #generating function of the time series
t_train = np.arange(0, 200, 0.5, dtype=float)
y_train_timeseries = ft_gen_ts(t_train)
t_test = np.arange(200, 400, 0.5, dtype=float)
y_test_timeseries = ft_gen_ts(t_test)
sample_length = 6
forecast_length = 400
X_train, y_train = build_samples(y_train_timeseries, sample_length)

model_org = build_lstm_model(sample_length)
model_org.summary()

batch_size = 50
epochs = 80
loss=losses.MeanSquaredError(reduction='auto', name='mean_squared_error')
optimizer=optimizers.Adam()

model_org.compile(loss=loss, optimizer=optimizer)

model_org.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=0)

y_forecast_org = compute_forecast(model_org)
error_org = loss(y_test_timeseries, y_forecast_org).numpy()

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
    y_forecast_org, error_org)
ai_org.print()
attempt_infos.append(ai_org)

end_step = np.ceil(len(X_train) / batch_size).astype(np.int32) * epochs

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
    model_pruning.fit(X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_pruning,
        verbose=0)
    model_pruned = extract_pruned_model(model_pruning)
    num_of_w_pruned, num_of_nz_w_pruned, num_of_z_w_pruned = \
        inspect_weigths(ac.name, model_pruned)
    y_forecast_pruned = compute_forecast(model_pruned)
    error_pruned = loss(y_test_timeseries, y_forecast_pruned).numpy()
    unzippedh5_size_pruned, zippedh5_size_pruned = retrieve_size_of_model(model_pruned)
    unzippedlt_size_pruned, zippedlt_size_pruned = retrieve_size_of_lite_model(model_pruned)

    mi_pruned = AttemptInfo (
        ac.name,
        num_of_w_pruned, num_of_nz_w_pruned, num_of_z_w_pruned,
        unzippedh5_size_pruned, zippedh5_size_pruned,
        unzippedlt_size_pruned, zippedlt_size_pruned,
        y_forecast_pruned, error_pruned)
    mi_pruned.print()
    attempt_infos.append(mi_pruned)

print('')
print('*** Final recap ***')
print_attempt_infos(attempt_infos)
scatter_attempt_infos(attempt_infos, t_test, y_test_timeseries)
