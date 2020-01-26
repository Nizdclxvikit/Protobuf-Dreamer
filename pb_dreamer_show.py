# Adapted by github.com/jnordberg from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream
# Adapted by github.com/ProGamerGov from https://github.com/jnordberg/dreamcanvas
# wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
# unzip -d model inception5h.zip

import argparse
import scipy.ndimage as spi
#from skimage.io import imread,imsave
import numpy as np
import os
import sys
import tensorflow as tf
import time

from io import BytesIO
from PIL import Image
from imageio import imwrite
import image_viewer


class Dreamer:
    def __init__(self, tile_size=512, print_model=True, verbose=True, initialize=True, display_path=None):
        self.tile_size = tile_size
        self.verbose = verbose
        self.window=None
        self.preview_size = (0,0)
        if initialize:
            model_path = "../../../models/tensorflow/inception5h/tensorflow_inception_graph.pb"

            self.layout = None


            #self.display_image = np.random.rand(640,640,3)
            #self.display_path = display_path

            # creating TensorFlow session and load model
            self.graph = tf.Graph()
            self.sess = tf.InteractiveSession(graph=self.graph)
            model_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
            with tf.gfile.FastGFile(model_fn, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            self.t_input = tf.placeholder(np.float32, name='input') # define the input tensor
            imagenet_mean = 117.0
            t_preprocessed = tf.expand_dims(self.t_input-imagenet_mean, 0)
            tf.import_graph_def(graph_def, {'input':t_preprocessed})

            self.resize = self.tffunc(np.float32, np.int32)(self.resize)




            # Optionally print the inputs and layers of the specified graph.
            if not print_model:
                print(self.graph.get_operations())
        else:

            #self.display_image = np.zeros((640,640,3), np.float32)
            self.layout = None
            self.graph = None
            self.sess = None
            self.t_input = None
            self.resize = None

    def T(self, layer):
        '''Helper for getting layer output tensor'''
        return self.graph.get_tensor_by_name("import/%s:0"%layer)

    def tffunc(self, *argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    # Helper function that uses TF to resize an image
    def resize(self, img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]


    def calc_grad_tiled(self, img, t_grad):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = self.tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = self.sess.run(t_grad, {self.t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def render_deepdream(self, t_grad, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        # split the image into a number of octaves
        img = np.float32(img0)[:,:,:3]
        octaves = []
        for i in range(octave_n-1):
            hw = img.shape[:2]
            lo = self.resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-self.resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = self.resize(img, hi.shape[:2])+hi
            for i in range(iter_n):
                #g = self.calc_grad_tiled(img, t_grad)
                g = self.calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))
                self.window.updateArray(img)
                if not self.verbose:
                    print ("Iteration Number: %d" % i)
            if not self.verbose:
                print ("Octave Number: %d" % octave)


        return np.uint8(np.clip(img/255.0, 0, 1)*255)

    #last_layer = None
    #last_grad = None
    #last_channel = None
    def render(self, img, layer='mixed4d_3x3_bottleneck_pre_relu', channel=-1, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4, preview_size=640):
        #global last_layer, last_grad, last_channel
        #if last_layer == layer and last_channel == channel:
        #    t_grad = last_grad
        #else:
        if channel == -1:
            t_obj = tf.square(self.T(layer))
        else:
            t_obj = self.T(layer)[:,:,:,channel]
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, self.t_input)[0] # behold the power of automatic differentiation!
        #last_layer = layer
        #last_grad = t_grad
        #last_channel = channel
        #end else
        img0 = np.float32(img)
        print(img0.shape)
        if preview_size != 0:
            self.preview_size = (preview_size, preview_size*img0.shape[0]//img0.shape[1])
        else:
            self.preview_size = (img0.shape[1],img0.shape[0])

        self.window = image_viewer.IV(name="DeepDream",size=self.preview_size)
        output = self.render_deepdream(t_grad, img0, iter_n, step, octave_n, octave_scale)
        self.window.close()
        return output

    def save(self, path, img):
        imwrite(path, np.uint8(np.clip(img/255.0, 0, 1)*255))

    def arrayToBytes(self, array):
        bytes = BytesIO()
        Image.fromarray(np.uint8(np.clip(array/255.0, 0, 1)*255)).save(bytes, format="PNG")
        bytes=bytes.getvalue()
        return bytes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, help="The input image for DeepDream. Ex: input.png", required=True)
    parser.add_argument('--output_image', '-o', default='output.png', help="The name of your output image. Ex: output.png", type=str)
    parser.add_argument('--channel', '-c', default='-1', help="The target channel of your chosen layer.", type=int)
    parser.add_argument('--layer', '-l', default='mixed4d_3x3_bottleneck_pre_relu', help="The name of the target layer.", type=str)
    parser.add_argument('--iter', '-itr', default='10', help="The number of iterations", type=int)
    parser.add_argument('--octaves', '-oct', default='4', help="The number of octaves.", type=int)
    parser.add_argument('--octave_scale', '-octs', default='1.4', help="The step size.", type=float)
    parser.add_argument('--step_size', '-ss', default='1.5', help="The step size.", type=float)
    parser.add_argument('--tile_size', '-ts', default='512', help="The size of your tiles.", type=int)
    parser.add_argument('--display_path', '-d', default=None, help="Path to display image to see changes live.", type=str)
    parser.add_argument('--print_model', help="Print the layers and inputs from the model.", action='store_false')
    parser.add_argument('--verbose', '-v', help="Prints the current iteration and current octave whenever either changes.", action='store_false')
    parser.add_argument('--preview_size', '-p', default='640', help="Horizontal dimension of the preview window.", type=int)
    parser.parse_args()
    args = parser.parse_args()
    input_img = args.input_image
    output_name = args.output_image
    channel_value = args.channel
    layer_name = args.layer
    iter_value = args.iter
    octave_value = args.octaves
    octave_scale_value = args.octave_scale
    step_size = args.step_size
    tile_size = args.tile_size
    print_model = args.print_model
    verbose = args.verbose
    display_path = args.display_path
    preview_size = args.preview_size
    input_img = Image.open(input_img)

    dreamer = Dreamer(tile_size=tile_size, print_model=print_model, verbose=verbose, display_path=display_path)

    output_img = dreamer.render(input_img, layer=layer_name, channel=channel_value, iter_n=iter_value, step=step_size, octave_n=octave_value, octave_scale=octave_scale_value, preview_size=preview_size)
    imwrite(output_name, output_img)
