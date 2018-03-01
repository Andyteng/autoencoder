import numpy as np
import struct
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist

#this is the size of our encoded represenations
encoding_dim = 32 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats(28*28)
    
#this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation = 'sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)
#create a placeholder for an encoded (32-dimentional) input
encoded_input = Input(shape=(encoding_dim,))
#retrive the last layer of the antoencoder model
decoder_layer = autoencoder.layers[-1]
#create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
#configure our model to use a pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss ='binary_crossentropy')

    
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)


#train our autoencoder for 25 epochs
autoencoder.fit(x_train, x_train,
                epochs=25,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test,x_test))

#Try to visualize the reconstructed inputs and the encoded representations
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10 # how many digits we will display
plt.figure(figsize=(20,4))
for i in range(n):
    #display original
    ax = plt.subplot(2, n,  i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

'''
def run():
    train_images = load_train_images() #(num_rows*num_cols,num_images)
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()

    #for i in range(10):
        #print(train_labels[i])
        #print(train_images.shape)
        #plt.imshow(train_images[:,i].reshape(28,28), cmap='gray')
        #plt.show()
        
 

    print ('done')

if __name__ == '__main__':
    run()
'''
