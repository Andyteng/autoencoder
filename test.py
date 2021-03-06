import numpy as np
import struct
import matplotlib.pyplot as plt

from keras.layers import Input, Dense
from keras.models import Model

train_images_idx3_ubyte_file = '/Users/Andyteng/Graduated_project/autoencoder/MNIST/train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = '/Users/Andyteng/Graduated_project/autoencoder/MNIST/train-labels-idx1-ubyte'
test_images_idx3_ubyte_file = '/Users/Andyteng/Graduated_project/autoencoder/MNIST/t10k-images-idx3-ubyte'
test_labels_idx1_ubyte_file = '/Users/Andyteng/Graduated_project/autoencoder/MNIST/t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print ('dimension: %d, Picture num: %d, picture size: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('already %d' % (i + 1) + ' pictures')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images.T


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print ('dimension:%d, picture num: %d' % (magic_number, num_images))

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('already %d' % (i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    """
    return decode_idx1_ubyte(idx_ubyte_file)




def run():
    train_images = load_train_images() #(num_rows*num_cols,num_images)
    train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()
    
    #this is the size of our encoded represenations
    encoding_dim = 32 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats(28*28)

    #for i in range(10):
        #print(train_labels[i])
        #print(train_images.shape)
        #plt.imshow(train_images[:,i].reshape(28,28), cmap='gray')
        #plt.show()
        
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
    decoder_layer = antocoder.layers[-1]
    #create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    #configure our model to use a pixel binary crossentropy loss, and the Adadelta optimizer
    autoencoder.compile(optimizer='adadelta', loss ='binary_crossentropy')

    print ('done')

if __name__ == '__main__':
    run()
