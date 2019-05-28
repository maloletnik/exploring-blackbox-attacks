import keras
from keras import backend as K
from keras.models import save_model
import tensorflow as tf

from tf_utils import tf_train, tf_test_error_rate
from mnist import *

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def main(model_name, model_type):
    np.random.seed(0)
    assert keras.backend.backend() == "tensorflow"
    set_mnist_flags()

    with tf.device('/gpu:0'):
        # flags.DEFINE_bool('NUM_EPOCHS', args.epochs, 'Number of epochs')
        # flags.DEFINE_integer('NUM_EPOCHS', args.epochs, 'Number of epochs')
        flags.DEFINE_integer('NUM_EPOCHS', 6, 'Number of epochs')

        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist()

        data_gen = data_gen_mnist(X_train)

        x = K.placeholder((None,
                           FLAGS.IMAGE_ROWS,
                           FLAGS.IMAGE_COLS,
                           FLAGS.NUM_CHANNELS
                           ))

        y = K.placeholder(shape=(None, FLAGS.NUM_CLASSES))

        model = model_mnist(type=model_type)

        # print(model.summary())

        # Train an MNIST model
        tf_train(x, y, model, X_train, Y_train, data_gen, None, None)

        # Finally print the result!
        _, _, test_error = tf_test_error_rate(model, x, X_test, Y_test)
        print('Test error: %.1f%%' % test_error)
        save_model(model, model_name)
        json_string = model.to_json()
        with open(model_name+'.json', 'wr') as f:
            f.write(json_string)


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("model", help="path to model")
    # parser.add_argument("--type", type=int, help="model type", default=1)
    # parser.add_argument("--epochs", type=int, default=6, help="number of epochs")
    # args = parser.parse_args()

    # main(args.model, args.type)

    # STANDARD TRAINING
    # main('models/modelA', 0)
    # main(model_name='models/modelB', model_type=1)
    # main(model_name='models/modelC', model_type=2)
    main(model_name='models/modelD', model_type=3)

