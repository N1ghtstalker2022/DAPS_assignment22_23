"""Data inference procedure.

This python file trains models to predict the closing stock price on each day for the data acquired, stored,
preprocessed and explored from previous steps. The data spans from April 2017 to April 2022. One single LSTM-based
model architecture is used for developing two separate models. One is a model for predicting the closing stock price
on each day for a 1-month time window (until end of May 2022), using only time series of stock prices. The other is A
model for predicting the closing stock price on each day for a 1-month time window (until end of May 2022),
using the time series of stock prices and the auxiliary data you collected. Furthermore, it evaluates the performance
of the model using mean absolute error and create visualizations to provide useful insight of the prediction result.

Typical usage example:

    infer(preprocessed_data_list)

"""
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from src.utils import create_dir

MAX_EPOCHS = 20
OUT_STEPS = 30


def infer(df):
    """Receive preprocessed data and construct two different experimental datasets.

    Conduct a comparison between prediction using only stocks data to predict and prediction using both stocks and
    auxiliary data.

    Args:
        df: Dataframe list containing stocks, weather and covid dataframes.

    """
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
    """Infer using LSTM-based model.

    Predict the next 30 days of stocks close value by the known data of former 30 days.
    Create baseline model to give a decent ablation study.
    Create LSTM model and feed data to it to train, validate and predict.

    Args:
        df: Pandas dataframe.
        df_name: Specific description of specific dataframe.

    """
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
        def call(self, inputs, training=None, mask=None):
            return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

    last_baseline = MultiStepLastBaseline()
    last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanAbsoluteError()])



    multi_val_performance = dict()
    multi_performance = dict()

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

    #  plot learning curve
    acc = history.history['mean_absolute_error']
    val_acc = history.history['val_mean_absolute_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training MAE')
    plt.plot(val_acc, label='Validation MAE')
    plt.legend(loc='upper right')
    plt.title('Training and Validation MAE for ' + df_name + ' model')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss for ' + df_name + ' model')
    fig.tight_layout()
    plt.savefig('infer/learning_curve_' + df_name + '_.png')
    plt.close()

    # test
    pred_input = tf.reshape(pred_df[0:30], [1, 30, num_features])
    pred_val = multi_lstm_model.predict(pred_input)
    pred_val = tf.squeeze(pred_val).numpy()

    compare_df = pred_df.copy()
    compare_df = compare_df[30:]
    compare_df.loc[:, 'Pred_Close'] = pred_val
    true_val = compare_df['Close']

    create_joint_plot(compare_df, 'Pred_Close', 'Close', df_name)
    create_residual_plot(pred_val, true_val, df_name)
    print("finish testing")
    # predict 05/01 to 05/31

    pred_input = tf.reshape(pred_df[30:], [1, 30, num_features])
    pred_val = multi_lstm_model.predict(pred_input)
    pred_val = tf.squeeze(pred_val).numpy()
    new_df = df.copy()
    extra_df = df.iloc[:1].copy()
    # cases should be zero before there is a statistics about covid
    for column in extra_df.columns:
        extra_df.loc[:, column] = 0
    cur_date_string = '2022-05-01'
    cur_date = datetime.datetime.strptime('2022-05-01', '%Y-%m-%d').date()
    pred_end_date = '2022-05-30'
    extra_df = extra_df.rename(index={df.iloc[:1].index[0]: cur_date}, inplace=False)
    pred_end_date = datetime.datetime.strptime(pred_end_date, '%Y-%m-%d').date()
    day_count = 0
    while cur_date <= pred_end_date:
        pred_single_val = pred_val[day_count]
        day_count = day_count + 1
        new_df = pd.concat([extra_df, new_df.loc[:]])
        new_df.loc[cur_date, 'Close'] = pred_single_val

        extra_df = extra_df.rename(
            index={cur_date: (cur_date + datetime.timedelta(days=1))},
            inplace=False)
        cur_date = cur_date + datetime.timedelta(days=1)
    new_df = new_df.set_index(new_df.index.astype(dtype='datetime64'))
    new_df = new_df.sort_index()

    show_stocks_with_pred(new_df, 'Close', df_name)
    print("finish pred")

def show_stocks_with_pred(df, column, name):
    """Plot line chart for stocks data and save in local disk.

    Args:
        stocks_df: Pandas dataframe for stocks data.
        column: Specific feature chosen from dataframe.

    """
    plt.figure()
    label = column + ' value'
    start_date = datetime.datetime.strptime('2017-04-01', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2022-04-30', '%Y-%m-%d')
    plt.plot(df[column].loc[start_date:end_date], label=label,
             linestyle='-', c='g')
    pred_start_date = datetime.datetime.strptime('2022-04-30', '%Y-%m-%d')
    pred_end_date = datetime.datetime.strptime('2022-05-30', '%Y-%m-%d')
    pre_label = 'predicted_' + column + ' value'
    plt.plot(df[column].loc[pred_start_date:pred_end_date], label=pre_label,
             linestyle='-', c='b')
    plt.xlabel('DateTime')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.savefig('infer/stock_' + column.lower() + '_with_pred_' + name + '.png')
    plt.close()

def create_residual_plot(predicted_val, true_val, name):
    """Create and save residual distribution for prediction and true value.

    Args:
        predicted_val: Predicted value by the machine learning model
        true_val: True value collected.
        name: Specific description of different model name

    Returns:

    """
    plt.figure()
    x = predicted_val
    y = true_val - predicted_val
    plt.scatter(x, y, c='blue')
    plt.savefig('infer/residual_distribution_' + name + '.png')
    plt.close()


def compile_and_fit(model, window, patience=2):
    """Compile the defined machine learning model and feed the training and validation data into the model.

    Args:
        model: Machine learning model applied.
        window: Window contains a certain length of data
        patience: Number of epochs with no improvement after which training will be stopped, used for early stopping.

    Returns:
        A History object. Its history attribute is a record of training loss values and metrics values
        at successive epochs, as well as validation loss values and validation metrics values (if applicable).
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    training_set = window.train
    validation_set = window.val
    history = model.fit(training_set, epochs=MAX_EPOCHS,
                        validation_data=validation_set, callbacks=[early_stopping])

    return history


def create_joint_plot(forecast, pred_val, true_val, title=None):
    """Create and save joint plot showing marginal distributions to understand the correlation between actual and
    predicted values

    Args:
        forecast: Pandas dataframe containing predicted values.
        pred_val: Feature representing prediction values.
        true_val: Feature representing true values.
        title: Words used to help define title of plotted figures.

    """
    g = sns.jointplot(x=pred_val, y=true_val, data=forecast, kind="reg", color="b")
    g.fig.set_figwidth(10)
    g.fig.set_figheight(10)

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
    [label.set_fontsize(13) for label in ax.xaxis.get_ticklabels()]
    [label.set_fontsize(13) for label in ax.yaxis.get_ticklabels()]

    ax.grid(ls=':')
    fig = g.fig
    fig.savefig('infer/pred_joint_' + title + '.png')


class WindowGenerator(object):
    """The window contains a certain number of data.

    Attributes:
        train_df: Training sets in Pandas dataframe format.
        val_df: Validation sets in Pandas dataframe format.
        test_df: Testing sets in Pandas dataframe format.
        _example: Example data used to give a visualization of model performance.
        label_columns: The feature that is determined to be predicted.
        column_indices: Dictionary mapping the number representing locations of features to features
        input_width: Designated input length of data.
        label_width: Designated input length of label.
        shift: Span of prediction.
        total_window_size: The total length of a window.
        input_slice: Create a slice object. This is used for input slicing.
        input_indices: Indices of input data for a given window.
        label_start: Start position for labels for a given window.
        labels_slice: Create a slice object. This is used for labels slicing.
        label_indices: Indices of labels data for a given window.

    """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self._example = None
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

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
        """Split window into inputs and labels windows."""
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
        """Give a visualization for prediction results."""
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
                predictions = model.predict(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [d]')

    def make_dataset(self, data):
        """Construct dataset in order to input into tensorflow model."""
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
