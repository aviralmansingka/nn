from keras.models import Sequential
from keras.layers import Dense, Activation

from keras.datasets import mnist
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 12


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32').reshape(-1, 784)
X_test = X_test.astype('float32').reshape(-1, 784)

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



model = Sequential()

model.add(Dense(10, input_dim=784, init='uniform', activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test Accuracy:', score[1])
