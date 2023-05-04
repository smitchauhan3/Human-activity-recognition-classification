import numpy as np
import os

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

DATA_PATH = os.path.join('Activity_Data')
actions = np.array(['clapping', 'hands_up', 'punching', 'standing', 'waving','walking','sitting'])
no_sequences = 30
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
#print(X.shape)
y = to_categorical(labels).astype(int)
#print(y.shape)
#y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#print(X_train.shape)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=700, callbacks=[tb_callback])
model.save('20-3_model.h5')
model.summary()

yhat = model.predict(X_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
print(accuracy_score(ytrue, yhat))
