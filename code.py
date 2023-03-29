import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import iencoder as ise
from itertools import cycle
classes = ['BENIGN',
           'DoS GoldenEye',
           'DoS Hulk',
           'Bot',
           'DoS Slowhttptest',
           'DoS slowloris',
           'Heartbleed',
           'Web Attack � Brute Force',
           'Web Attack � Sql Injection',
           'Web Attack � XSS',
           'DDoS',
           'PortScan',
           'Infiltration',
           'FTP-Patator',
           'SSH-Patator']
def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
def plot_loss_acc(history):
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(history.history['loss'][1:])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('train loss')
    plt.subplot(122)
    plt.plot(history.history['val_loss'][1:])
    plt.xlabel('epoch')
    plt.title('val loss')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.tight_layout()
    plt.show()
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(history.history['accuracy'][1:])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('train accuracy')
    plt.subplot(122)
    plt.plot(history.history['val_accuracy'][1:])
    plt.xlabel('epoch')
    plt.title('val accuracy')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.tight_layout()
    plt.show()
def normalization(data):
    mmax = np.max(data, axis=0)
    mmin = np.min(data, axis=0)
    data_norm = (data - mmin) / (mmax - mmin + 1e-3)
    return data_norm
def draw_cm(y_test, y_pred):
    y_true = [item.tolist().index(1) for item in y_test]
    labels = list(range(15))
    y_pred = [np.argmax(item) for item in y_pred]
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()
def draw_roc(y_test, y_pred):
    fpr, tpr, roc_auc = {}, {}, {}
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
    if y_test.shape[1] > 2:
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'purple', 'blue'])
        for i in range(y_test.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i, color in zip(range(y_test.shape[1]), colors):
            name = classes[i]
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='{0} (area = {1:0.2f})'
                                                              ''.format(name, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
def draw_cm_roc(y_test, y_pred):
    draw_cm(y_test, y_pred)
    draw_roc(y_test, y_pred)
def pre_process(file_dir, num=1000):
    datas, count = {}, {}
    files = [os.path.join(file_dir, item) for item in os.listdir(file_dir)]
    tbar = tqdm(files, desc='data pre_processing')
    for file in tbar:
        data = pd.read_csv(file, encoding="utf-8")
        data.dropna(inplace=True)
        grouped = data.groupby(" Label")
        for name, group in grouped:
            group = np.array(group)
            if name not in datas:
                datas[name] = group
                count[name] = group.shape[0]
            else:
                datas[name] = np.vstack([datas[name], group])
                count[name] += group.shape[0]

        tbar.set_description('pre_processing:[{}]'.format(file))
    data = [item for item in datas.values()]
    data = np.concatenate(data)
    np.random.shuffle(data)
    features = data[:, :-1]
    labels = data[:, -1]
    features_selected, labels_selected = [], []
    temp_dict = {}
    for feature, label in zip(features, labels):
        if label in temp_dict:
            temp_dict[label] += 1
        else:
            temp_dict[label] = 1
        if temp_dict[label] < num + 1:
            features_selected.append(feature)
            labels_selected.append(label)
    features_selected = np.vstack(features_selected)
    labels_selected = np.vstack(labels_selected)
    x = normalization(features_selected)
    x = np.clip(x, 0, 1)
    y = np.array([classes.index(item) for item in labels_selected]).reshape(-1, 1)
    return x, y
def get_dataloader(config):
    file_dir = config['file_dir']
    num = config['num']
    num_classes = config['num_classes']
    x, y = pre_process(file_dir, num=num)
    unique, count = np.unique(y, return_counts=True)
    data_count = dict(zip(unique, count))
    data_count_train = dict(zip(data_count, map(lambda x: int(x * 0.7), data_count.values())))
    class_weight = dict(zip(data_count_train, map(lambda x: float(1 / x), data_count.values())))
    data = np.hstack([x, y])
    data = data[data[:, -1].argsort()]
    train, val, test = [], [], []
    start = 0
    for k, v in data_count.items():
        train.append(data[start:start + int(v * 0.7)])
        val.append(data[start + int(v * 0.7):start + int(v * 0.9)])
        test.append(data[start + int(v * 0.9):start + v])
        start += v
    train = np.concatenate(train)
    val = np.concatenate(val)
    test = np.concatenate(test)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_val = val[:, :-1]
    y_val = val[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]
    y_train_onehot = to_categorical(y_train, num_classes)
    y_val_onehot = to_categorical(y_val, num_classes)
    y_test_onehot = to_categorical(y_test, num_classes)
    return x_train, y_train_onehot, x_val, y_val_onehot, x_test, y_test_onehot, class_weight
K.clear_session()
class AMLP(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        n_features = config['n_features']
        num_classes = config['num_classes']
        start_neurons = config['start_neurons']
        dropout_rate = config['dropout_rate']
        activation = config['activation']
        self.model = Sequential([
            ise(start_neurons, input_shape=(n_features,))
            Dense(start_neurons, input_shape=(n_features,), activation=activation),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(start_neurons // 2, activation=activation),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(start_neurons // 4, activation=activation),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(start_neurons // 8, activation=activation),
            BatchNormalization(),
            Dropout(dropout_rate / 2),
            Dense(num_classes, activation='softmax')])
    def call(self, x):
        x = self.model(x)
        return x
def MLP_res(input_shape, dropout_rate=0.25, activation='relu'):
    start_neurons = 512
    X_input = Input(input_shape)
    x1 = Dense(start_neurons, input_dim=78, activation=activation)(X_input)
    x = BatchNormalization()(x1)
    x2 = Dropout(dropout_rate)(x)
    x = Add()([x1, x2])
    x1 = Dense(start_neurons // 2, activation=activation)(x)
    x = BatchNormalization()(x1)
    x2 = Dropout(dropout_rate)(x)
    x = Add()([x1, x2])
    x1 = Dense(start_neurons // 4, activation=activation)(x)
    x = BatchNormalization()(x1)
    x2 = Dropout(dropout_rate)(x)
    x = Add()([x1, x2])
    x1 = Dense(start_neurons // 8, activation=activation)(x)
    x = BatchNormalization()(x1)
    x2 = Dropout(dropout_rate / 2)(x)
    x = Dense(15, activation='softmax')(x2)
    model = Model(X_input,x, name= 'MLP_res')
    return model
class MLP(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.patience = config['patience']
        check_folder(config['save_path'])
        self.save_path = os.path.join(config['save_path'], 'weights_best.hdf5')
        self.model = AMLP(config)
    def train(self, x_train, y_train, x_val, y_val, class_weight):
        x_train, y_train = x_train.astype('float64'), y_train.astype('float64')
        x_val, y_val = x_val.astype('float64'), y_val.astype('float64')
        call_ES = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=self.patience,
                                verbose=1,
                                mode='auto',
                                baseline=None)
        checkpoint = ModelCheckpoint(filepath=self.save_path,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='auto',
                                     period=1)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x_train, y_train,
                                 validation_data=[x_val, y_val],
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 callbacks=[call_ES, checkpoint],
                                 shuffle=True,
                                 class_weight=class_weight,
                                 verbose=1)
        plot_loss_acc(history)
if __name__ == '__main__':
    config = {'n_features': 78,
              'num_classes': 15,
              'start_neurons': 512,
              'dropout_rate': 0.25,
              'activation': 'relu',
              'epochs': 100,
              'batch_size': 50,
              'patience': 50,
              'save_path': 'saved',
              'file_dir': 'datas',
              'num': 1000,
              }
    x_train, y_train, x_val, y_val, x_test, y_test, class_weight = get_dataloader(config)
    mlp = MLP(config)
    mlp.train(x_train, y_train, x_val, y_val, class_weight)
    x_test = x_test.astype('float64')
    y_pred, y_true = mlp.model.predict(x_test), y_test
    draw_cm_roc(y_true, y_pred)
