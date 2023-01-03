import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from src.utils import create_dir

MAX_EPOCHS = 20
OUT_STEPS = 30


def infer(df):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    create_dir('infer')
    stocks_data = df['stocks']
    weather_data = df['weather']
    covid_data = df['covid']

    df = stocks_data.drop('Volume', axis=1)

    integrated_df = df.merge(weather_data, left_index=True, right_index=True)
    integrated_df = integrated_df.merge(covid_data, left_index=True, right_index=True)

    # column_indices = {name: i for i, name in enumerate(df.columns)}

    # predicting the closing stock price on each day for a 1-month time window (until end of May 2022), using only
    # time series of stock prices.
    infer_by_features(df, 'stocks')
    # predicting the closing stock price on each day for a 1-month time window (until end of May 2022), using the
    # time series of stock prices and the auxiliary data you collected.
    infer_by_features(integrated_df, 'integrated')


def infer_by_features(df, df_name):
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]
    pred_df = df[n - 60:]

    num_features = df.shape[1]

    multi_window = WindowGenerator(input_width=30,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df,
                                   label_columns=['Close'])

    class MultiStepLastBaseline(tf.keras.Model):
        def call(self, inputs):
            return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

    last_baseline = MultiStepLastBaseline()
    last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanAbsoluteError()])

    multi_val_performance = {}
    multi_performance = {}

    multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
    multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)

    # modify plot name
    multi_window.plot(last_baseline)
    plt.savefig('infer/baseline_pred_' + df_name + '.png')
    plt.close()

    # lstm model

    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(OUT_STEPS * 1,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, 1])
    ])

    # train
    history = compile_and_fit(multi_lstm_model, multi_window)

    multi_lstm_model.summary()

    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)

    for name, value in multi_performance.items():
        print(f'{name:8s}: {value[1]:0.4f}')

    multi_window.plot(multi_lstm_model)
    plt.savefig('infer/lstm_pred_' + df_name + '.png')
    plt.close()

    # predict
    # multi_window.pred_df = pred_df
    # multi_window.label_columns = None
    # pred_data = multi_window.pred
    pred_input = tf.reshape(pred_df[0:30], [1, 30, num_features])
    true_val = pred_df[30:].loc[:, 'Close']
    pred_val = multi_lstm_model.predict(pred_input)
    pred_val = tf.squeeze(pred_val).numpy()

    compare_df = pred_df.copy()
    compare_df = compare_df[30:]
    compare_df.loc[:, 'Pred_Close'] = pred_val
    true_val = compare_df['Close']

    create_joint_plot(compare_df, 'Pred_Close', 'Close', df_name)
    create_residual_plot(pred_val, true_val, df_name)
    print("finish prediction")


def create_residual_plot(predicted_val, true_val, name):
    plt.figure()
    x = predicted_val
    y = true_val - predicted_val
    plt.scatter(x, y, c='blue')
    plt.savefig('infer/residual_distribution_' + name + '.png')
    plt.close()


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    training_set = window.train
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val, callbacks=[early_stopping])
    return history


def create_joint_plot(forecast, pred_val, true_val, title=None):
    g = sns.jointplot(x=pred_val, y=true_val, data=forecast, kind="reg", color="b")
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]
    if title is not None:
        ax.set_title(title, fontsize=16)

    ax = g.fig.axes[0]
    ax.text(0.1, 0.1, "R = {:+4.2f}".format(forecast.loc[:, [true_val, pred_val]].corr().iloc[0, 1]), fontsize=16)
    ax.set_xlabel('Predictions', fontsize=15)
    ax.set_ylabel('Observations', fontsize=15)
    # ax.set_xlim(0, 0.5)
    # ax.set_ylim(0, 0.5)
    ax.grid(ls=':')
    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

    ax.grid(ls=':')
    fig = g.fig
    fig.savefig('infer/pred_joint_' + title + '.png')


class WindowGenerator(object):
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, pred_df=None,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.pred_df = pred_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [d]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    @property
    def pred(self):
        return self.make_dataset(self.pred_df)

# if __name__ == "__main__":
#     df = preprocess_stocks_data()
#     infer(df)
