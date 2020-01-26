import PySimpleGUI as sg
import numpy as np
from PIL import Image
from io import BytesIO

class IV:
    def __init__(self, name="Image Viewer", size=(640,320)):
        self.method = Image.BILINEAR
        self.size = size
        self.image = IV.arrayToImage(np.zeros((self.size[-1::-1])))
        layout = [[sg.Image(data=IV.imageToBytes(self.image), key="image1")]]
        self.window = sg.Window(name, layout)
        self.window.read(timeout=0)


    @staticmethod
    def imageToBytes(image):
        bytes = BytesIO()
        image.save(bytes, format="PNG")
        bytes=bytes.getvalue()
        return bytes

    @staticmethod
    def arrayToBytes(array):
        return IV.imageToBytes(IV.arrayToImage(array))

    @staticmethod
    def imageToArray(image):
        return np.float32(image)[:,:,:3]

    @staticmethod
    def arrayToImage(array):
        return Image.fromarray(np.uint8(np.clip(array/255.0, 0, 1)*255))

    @staticmethod
    def resizeArray(array,size):
        return IV.imageToArray(self.resizeImage(IV.arrayToImage(array), size))

    def resizeImage(self, image, size):
        return image.resize(size, self.method)

    def setResizeMethod(self,method):
        self.method = method
        self.updateImage()

    def updateImage(self, newImage=None, useImageSize=False, size=None):
        if newImage:
            self.image = newImage

        if useImageSize:
            self.size = self.image.size
        elif size:
            self.size = size

        self.window["image1"].Update(data=IV.imageToBytes(self.resizeImage(self.image,self.size)))
        self.window.Refresh()

    def updateArray(self, newImage, useImageSize=False, size=None):
        self.updateImage(IV.arrayToImage(newImage), useImageSize, size)

    def close(self):
        self.window.close()

    def open(self):
        self.window = sg.Window(name, layout)
        self.window.read(timeout=0)
