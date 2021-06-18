import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist

def plot_loss_curve(history):

    plt.figure(figsize=(15, 10))

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()   

def plot_accuracy_curve(history):

    plt.figure(figsize=(15, 10))

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()   

def train_mnist_model():
    X, y = [], []
    for i in range(2000):
        X.append(cv2.imread('images/food/food%d.jpg'%(i+1)))
        y.append(0)
    print('food insert finished')
    
    for i in range(1500):
        X.append(cv2.imread('images/interior/interior%d.jpg'%(i+1)))
        y.append(1)
    print('interior insert finished')
    
    for i in range(1000):
        X.append(cv2.imread('images/exterior/exterior%d.jpg'%(i+1)))
        y.append(2)
    print('exterior insert finished')
    
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    model = Sequential([
                Input(shape=(300,300,3), name='input_layer'),
                Conv2D(32, kernel_size=3, activation='relu', name='conv_layer1'),
                MaxPooling2D(pool_size=2),
                Conv2D(64, kernel_size=3, activation='relu', name='conv_layer2'),
                MaxPooling2D(pool_size=2),
                Dropout(0.5),
                Flatten(),
                Dense(64, activation='relu', name='output_layer1'),
                Dense(3, activation='softmax', name='output_layer2')
            ])

    model.summary()    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=10)
    plot_loss_curve(history.history)
    plot_accuracy_curve(history.history)
    print(history.history) 
    
    model.save('model-201711809')
    
    return model
    
def insert_sample_data():
    X = []
    y = [0,0,0,0,1,1,1,2,2,2]
    
    for i in range(10):
        X.append(cv2.imread('images/sample/sample%d.jpg'%(i+1)))
    print('sample insert finished')
    
    X = np.array(X)
    y = np.array(y)

    return X, y

def predict_image_sample(model, X_test, y_test):

    pred = []
    for i in range(10):
        test_image = X_test[i]
        test_image = test_image.reshape(1,300,300,3)

        y_actual = y_test[i]
        print("y_actual number=", y_actual)
        
        y_pred = model.predict(test_image)
        y_pred = np.argmax(y_pred, axis=1)[0]
        print("y_pred=", y_pred)
        pred.append(y_pred)

    # print(accuracy_score(y_test, pred))
    # print(recall_score(y_test, pred))
    # print(precision_score(y_test, pred))
    # print(f1_score(y_test, pred)) 

if __name__ == '__main__':
    # train_mnist_model()
    model = load_model('model-201711809')
    X_test, y_test = insert_sample_data()
    predict_image_sample(model, X_test, y_test)