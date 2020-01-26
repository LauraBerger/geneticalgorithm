import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cv2
from termcolor import colored
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cv2
import numpy as np
import pydicom
import os
from PIL import Image
import matplotlib as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt



class Image:

    def __init__(self, PathDicom, grayscale,imagenumber):



        #self.image = self.load_image_file(PathDicom,grayscale)



        self.PathDicom = PathDicom
        self.lstFilesDCM = []
        self.imagenummber = imagenumber
        #self.lstFilesDCM=self.load_Image_files(grayscale)
        #Dataset = self.load_Image(grayscale,imagenumber)
        self.image = self.load_Image(grayscale, imagenumber)

        self.row = -1
        self.col = -1
        self.channel = -1
        #self.resizeImage = self.resize_image(256)


        self.rotation_angle = math.radians(0.0)
        self.shear_vertical = math.radians(0.0)
        self.shear_horizontal = math.radians(0.0)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.trans_x = 0.0
        self.trans_y = 0.0
    def histogrammstauchung(self):
        PixelArray=self.pixelvalues_in_array()
        pixmax=PixelArray.max()
        pixmin=PixelArray.min()
        for m in range (len(PixelArray)):
            for n in range(len(PixelArray)):
              pix=PixelArray[m,n]
              pixstretch=255*((pix-pixmin)/(pixmax-pixmin))
              self.image=pixstretch


    def load_Image(self,as_gray: bool, imagenumber):
        self.lstFilesDCM=self.load_Image_files(as_gray)

        for dirName, subdirList, fileList in os.walk(self.PathDicom):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    self.lstFilesDCM.remove('/Users/macbookair/Documents/Internship/Image Datasets/test/data/.DS_Store')
                    image=self.pixelvalues_in_array()
                    return image
                    break

                elif ".png" in filename.lower():
                    self.lstFilesDCM.remove('./images/test/.DS_Store')
                    image = self.load_image_file(as_gray, imagenumber)
                    #print("Image in load Image: ", image)
                    self.image=image
                    return image
                    break

                elif ".jpeg" in filename.lower():
                    self.lstFilesDCM.remove('./images/test/.DS_Store')
                    image = self.load_image_file(as_gray, imagenumber)
                    #print("Image in load Image: ",image)
                    self.image=image
                    return image
                    break

    def load_Image_files(self,as_gray: bool):

        for dirName, subdirList, fileList in os.walk(self.PathDicom):
            for filename in fileList:
                self.lstFilesDCM.append(os.path.join(dirName, filename))
        #print("Liste neu",self.lstFilesDCM)
        return self.lstFilesDCM





    def load_image_file(self , as_gray: bool,imagenumber):
        print(self.lstFilesDCM)
        #self.lstFilesDCM[0]
        #lstFilesDCM=[]
        if as_gray == True:
            #print('if: ',self.lstFilesDCM[imagenumber])
            image = cv2.imread(self.lstFilesDCM [imagenumber],cv2.IMREAD_GRAYSCALE)
            #print("image",image)

            #self.row, self.col = image.shape
        else:
            image = cv2.imread(self.lstFilesDCM [imagenumber], cv2.IMREAD_UNCHANGED)
            #image = cv2.cvtColor(self.lstFilesDCM [0], cv2.COLOR_BGRA2RGBA)
            #print("image",image)
            #self.row, self.col, self.channel = image.shape

        return image


    def get_image(self):
        #print("self.image   ",self.image)
        return self.image

    def pixelvalues_in_array(self):
        ds = self.get_file()
        PixelArray,ds = self.get_pixel_Values(ds)
        #print("Slice Loacation",ds.BitsStored)
        #print("Pixel Maximum",PixelArray.max())
        PixelArraymax=4096/255
        #print("neuer teilwert:",PixelArraymax)
        PixelArray=PixelArray/PixelArraymax
        #print("Pixel Maximum", PixelArray.max())
        #print(PixelArray)
        return PixelArray

    def get_file(self):  # gibt die Bildinformationen zururck
        lstFilesDCM = self.lstFilesDCM
        n=self.imagenummber
        #print(n)
        return pydicom.read_file(lstFilesDCM[n])

    def get_pixel_Values(self,ds):
        return ds.pixel_array,ds


    def resize_image(self, size):
        #print(size)
        #print(self.image.shape)

        #print (colored(self.image, 'red'))
        self.row,self.col=self.image.shape
        #print(colored(self.row,'green'))
        data_downsampling = 0
        if self.row == self.col:
            if self.row != size:
                print('The image has {} x {} voxels'.format(self.row, self.col))  # ds.pixel_array.shape
            if self.row> size:
                data_downsampling = self.image[::2, ::2]
                #print('The downsampled image has {} x {} voxels'.format( data_downsampling.shape[0], data_downsampling.shape[1]))
        return data_downsampling
        #self.image = cv2.resize(self.image, size)
        #self.row, self.col = self.image.shape




    def show_image(self):
        plt.imshow(self.image, cmap="gray")
        plt.axis('on')
        plt.show()

    def show_image2(self, image):
        plt.imshow(image, cmap="gray")
        plt.axis('on')
        plt.show()

    def transform_image_par(self, trans: (), scale: (), rotation: float, shear: ()):
        trans_m = self.calc_affine_transformation_matrix(trans, scale, rotation, shear)
        #self.image= transform.warp(self.image, trans_m)
        print(self.image)
        self.image = cv2.warpAffine(self.image, trans_m[:2, :], self.image.shape, flags=cv2.INTER_LINEAR)

    def transform_image_matrix(self, trans_m):
        return cv2.warpAffine(self.image, trans_m[:2, :], self.image.shape, flags=cv2.INTER_LINEAR)

    def calc_transform_image(self, input):
        trans = input[:2]
        scale = input[2:4]
        rotation = input[4]
        shear = input[5:7]

        trans_m = self.calc_affine_transformation_matrix(trans, scale, rotation, shear)

        # warp = transform.warp(self.image, trans_m)
        warp = cv2.warpAffine(self.image, trans_m[:2, :], self.image.shape, flags=cv2.INTER_LINEAR)
        # warp = cv2.warpAffine(self.image, trans_m[:2,:], self.image.shape, flags = cv2.INTER_CUBIC)

        return warp

    def calc_affine_transformation_matrix(self, trans, scale, rotation, shear):
        self.rotation_angle = math.radians(rotation)
        self.shear_vertical = math.radians(shear[0])
        self.shear_horizontal = math.radians(shear[1])
        self.scale_x = scale[0]
        self.scale_y = scale[1]
        self.tx = trans[0]
        self.ty = trans[1]

        a0 = self.scale_x * math.cos(self.rotation_angle) - self.shear_vertical * self.scale_x * math.sin(
            self.rotation_angle)
        a1 = self.shear_vertical * self.scale_y * math.cos(self.rotation_angle) + self.scale_y * math.sin(
            self.rotation_angle)
        a2 = 0.5 * (-self.scale_x * math.cos(
            self.rotation_angle) * self.row + self.shear_vertical * self.scale_x * math.sin(
            self.rotation_angle) * self.row + self.row - self.col * self.shear_vertical * self.scale_y * math.cos(
            self.rotation_angle) + 2 * self.scale_x * self.tx * math.cos(
            self.rotation_angle) + 2 * self.shear_vertical * self.scale_y * self.ty * math.cos(
            self.rotation_angle) - self.col * self.scale_y * math.sin(
            self.rotation_angle) - 2 * self.shear_vertical * self.scale_x * self.tx * math.sin(
            self.rotation_angle) + 2 * self.scale_y * self.ty * math.sin(self.rotation_angle))

        b0 = self.shear_horizontal * self.scale_x * math.cos(self.rotation_angle) - self.scale_x * math.sin(
            self.rotation_angle)
        b1 = self.scale_y * math.cos(self.rotation_angle) + self.shear_horizontal * self.scale_y * math.sin(
            self.rotation_angle)
        b2 = 0.5 * (-self.scale_y * math.cos(
            self.rotation_angle) * self.col - self.shear_horizontal * self.scale_y * math.sin(
            self.rotation_angle) * self.col + self.col - self.row * self.shear_horizontal * self.scale_x * math.cos(
            self.rotation_angle) + 2 * self.shear_horizontal * self.scale_x * self.tx * math.cos(
            self.rotation_angle) + 2 * self.scale_y * self.ty * math.cos(
            self.rotation_angle) + self.row * self.scale_x * math.sin(
            self.rotation_angle) - 2 * self.scale_x * self.tx * math.sin(
            self.rotation_angle) + 2 * self.shear_horizontal * self.scale_y * self.ty * math.sin(self.rotation_angle))

        c0 = 0
        c1 = 0
        c2 = 1

        trans_m = np.array([(a0, a1, a2), (b0, b1, b2), (c0, c1, c2)])
        # trans_m = np.linalg.inv(trans_m)

        return trans_m



