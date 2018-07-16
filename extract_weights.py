import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import mxnet as mx
import mxnet.ndarray as nd

from mylib.keras_deeplabv3p import Deeplabv3 as Deeplabv3p_keras
from mylib.keras_to_gluon import WeightConverter, set_gpu_usage
from mylib.deeplabv3p import DeepLabv3p as DeepLabv3p_gluon


def get_xception_filename(key):
    """Rename tensor name to the corresponding Keras layer weight name.
    # Arguments
        key: tensor name in TF (determined by tf.variable_scope)
    """
    filename = str(key)
    filename = filename.replace('/', '_')
    filename = filename.replace('xception_65_', '')
    filename = filename.replace('decoder_', '', 1)
    filename = filename.replace('BatchNorm', 'BN')
    if 'Momentum' in filename:
        return None
    if 'entry_flow' in filename or 'exit_flow' in filename:
        filename = filename.replace('_unit_1_xception_module', '')
    elif 'middle_flow' in filename:
        filename = filename.replace('_block1', '')
        filename = filename.replace('_xception_module', '')

    # from TF to Keras naming
    filename = filename.replace('_weights', '_kernel')
    filename = filename.replace('_biases', '_bias')

    return filename + '.npy'


def extract_tensors_from_checkpoint_file(filename, output_folder='weights'):
    """Extract tensors from a TF checkpoint file.
    # Arguments
        filename: TF checkpoint file
        output_folder: where to save the output numpy array files
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = tf.train.NewCheckpointReader(filename)

    for key in reader.get_variable_to_shape_map():
        # convert tensor name into the corresponding Keras layer weight name and save
        filename = get_xception_filename(key)
        if filename:
            path = os.path.join(output_folder, filename)
            arr = reader.get_tensor(key)
            np.save(path, arr)
            print("tensor_name: ", key, "shape: ", arr.shape)


def npy_to_keras(classes, load_from, save_to):
    print('Instantiating an empty Deeplabv3+ model...')
    model = Deeplabv3p_keras(input_shape=(512, 512, 3), classes=classes)

    print('Loading weights from', load_from)
    error = 0
    weight_files = set(os.listdir(load_from))

    for layer in tqdm(model.layers):
        if layer.weights:
            try:
                weights = []
                for w in layer.weights:
                    weight_name = os.path.basename(w.name).replace(':0', '')
                    weight_file = layer.name + '_' + weight_name + '.npy'
                    weight_files.remove(weight_file)
                    weight_arr = np.load(os.path.join(load_from, weight_file))
                    weights.append(weight_arr)
                layer.set_weights(weights)
            except Exception as e:
                raise e
                error += 1
                print("Fail to extract layer weights:", layer.name)
                print("Due to error:", e)

    print('Saving model weights... # error layers:', error)
    print("Unused: %s weights" % len(weight_files))
    for weight_file in weight_files:
        print(weight_file, np.load(os.path.join(load_from, weight_file)).shape)
    model.save_weights(save_to)
    return model


if __name__ == '__main__':
    '''Note: if there is gpu memory error, use cpu runtime instead.'''

    BASE = '/home/jiancheng/code/segmentation/deeplabv3p_gluon/tmp_weights'

    CKPT = "/home/jiancheng/code/segmentation/deeplabv3p_gluon/workspace/tmptf/deeplabv3_pascal_train_aug/model.ckpt"
    FLAG = "pascal_train_aug"
    CLASSES = 21

    # CKPT = "/home/jiancheng/code/segmentation/deeplabv3p_gluon/workspace/tmptf/deeplabv3_pascal_trainval/model.ckpt"
    # FLAG = "pascal_trainval"
    # CLASSES = 21

    # CKPT = "/home/jiancheng/code/segmentation/deeplabv3p_gluon/workspace/tmptf/deeplabv3_cityscapes_train/model.ckpt"
    # FLAG = "cityscapes_train"
    # CLASSES = 19
    #
    # CKPT = "/home/jiancheng/code/segmentation/deeplabv3p_gluon/workspace/tmptf/deeplabv3_xception_ade20k_train/model.ckpt"
    # FLAG = "ade20k_train"
    # CLASSES = 151
    #
    # CKPT = "/home/jiancheng/code/segmentation/deeplabv3p_gluon/workspace/tmptf/xception/model.ckpt"
    # FLAG = "imagenet_pretrain_for_pascal"
    # CLASSES = 21

    set_gpu_usage()

    output_folder = os.path.join(BASE, FLAG, "TF2NPY")
    extract_tensors_from_checkpoint_file(CKPT, output_folder=output_folder)

    keras_model = npy_to_keras(classes=CLASSES, load_from=output_folder,
                               save_to=os.path.join(BASE, FLAG, "%s.h5" % FLAG))

    print("Keras model setup. Output shape: ", keras_model.output.get_shape())

    weight_converter = WeightConverter(keras_model=keras_model)

    gluon_model = DeepLabv3p_gluon(classes=CLASSES)
    gluon_model.initialize(ctx=mx.gpu())
    inputs = nd.random_normal(shape=(1, 3, 512, 512), ctx=mx.gpu())
    outputs = gluon_model(inputs)
    print("Gluon model setup. Output shape: ", outputs.shape)

    weight_converter.set_parameters(gluon_model)

    gluon_model.save_params(os.path.join(BASE, FLAG, "%s.params" % FLAG))

    print("Clear.")
