from keras import models
from keras import optimizers
from numpy import *
from keras import layers
from keras.datasets import  imdb
def vectorize_sequences(sequences,dimension=10000):
    results=zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.0
    return results

(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)
y_train=asarray(train_labels).astype('float32')
y_test=asarray(test_labels).astype('float32')
model=models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=4,batch_size=512)
results=model.evaluate(x_test,y_test)
print(results)
print(model.predict(x_test))

