import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cv2

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



class Histogram:

    def calc_histogram(self, image):
        # histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
        histogram = cv2.calcHist([np.uint8(image)], [0], None, [256], [0, 256])
        bin_edges = np.arange(0, 257)

        return histogram, bin_edges[0:-1]

    def show_histogram(self, image):
        histogram = self.calc_histogram(image)

        plt.figure()
        plt.plot(histogram[1], histogram[0])
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixels")
        plt.show()

    def calc_joint_histogram(self, image_1, image_2):
        # joint_histogram, x_edges, y_edges = np.histogram2d(image_1.ravel(), image_2.ravel(), bins=256)
        joint_histogram = cv2.calcHist([np.uint8(image_1), np.uint8(image_2)], [0, 1], None, [256, 256],
                                       [0, 256, 0, 256])
        return joint_histogram

    def show_joint_histogram(self, image_1, image_2, log: bool):
        joint_histogram = self.calc_joint_histogram(image_1, image_2)
        plt.figure()

        if log:
            log_joint_histogram = np.zeros(joint_histogram.shape)
            non_zeros = joint_histogram != 0

            log_joint_histogram[non_zeros] = np.log(joint_histogram[non_zeros])
            # log_joint_histogram = 255*(log_joint_histogram/math.ceil(log_joint_histogram.max()))
            # print(log_joint_histogram.max())
            plt.imshow(log_joint_histogram.T, cmap="gray")
        else:
            plt.imshow(joint_histogram.T, cmap="gray")

        plt.xlabel("Image_1")
        plt.ylabel("Image_2")

        plt.show()

    def mutual_information(self, joint_histogram):
        pxy = joint_histogram / float(np.sum(joint_histogram))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0

        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

        return mi

    def normalized_mutal_information(self, joint_histogram):
        pxy = joint_histogram / float(np.sum(joint_histogram))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
        hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
        hy = -np.sum(py[py > 0] * np.log(py[py > 0]))

        nmi = hx + hy / hxy
        return nmi


class Fitness_Calc:

    def __init__(self, base_image: Image, trans_image: Image):
        self.base_image = base_image
        self.trans_image = trans_image

        self.base_image_data = base_image.get_image()
        self.trans_image_data = trans_image.get_image()

        self.histogram = Histogram()
        size = len(self.base_image_data)

        self.matrixsize = 1.0 / (size * size)
        self.matrixsize_plus = 1.0 / (size + size - 1)

        self.fitnessfunction="SSIM"

    def fitness(self, input_data):
        trans = input_data[:2]
        scale = input_data[2:4]
        rotation = input_data[4]
        shear = input_data[5:7]
        fitness=0
        trans_m = self.trans_image.calc_affine_transformation_matrix(trans, scale, rotation, shear)
        warp_i = self.trans_image.transform_image_matrix(trans_m)
        if "mutalinformation"==self.fitnessfunction:
            joint_histogram = self.histogram.calc_joint_histogram(self.base_image_data, warp_i)
            fitness = self.histogram.mutual_information(joint_histogram)
        elif"imageQuality"==self.fitnessfunction:
            #self.trans_image_data=warp_i
            fitness=self.image_Quality_Index(self.base_image_data,warp_i)
        elif "RMSE"==self.fitnessfunction:
                sumpix = ((self.base_image_data - warp_i) * (self.base_image_data - warp_i)).sum()
                result = math.sqrt(self.matrixsize * sumpix)
                fitness= 1 - result *(-100)
        elif"SSIM"==self.fitnessfunction:
            #self.trans_image_data = warp_i
            #s=S()
            #fitness=S.Sfunction(s,self.base_image_data,warp_i)
            fitness=self.SSIM(warp_i)
        else:
            print("No Fitness Function is chossen")
            print("Please choose a Fitness Function!")
            breakpoint()
        #print(fitness)
        return fitness

    def nmi_fitness(self, input_data):
        trans = input_data[:2]
        scale = input_data[2:4]
        rotation = input_data[4]
        shear = input_data[5:7]

        trans_m = self.trans_image.calc_affine_transformation_matrix(trans, scale, rotation, shear)
        warp_i = self.trans_image.transform_image_matrix(trans_m)
        joint_histogram = self.histogram.calc_joint_histogram(self.base_image_data, warp_i)

        nmi = self.histogram.normalized_mutal_information(joint_histogram)

        return nmi



    def get_firstsummand(self,baseimage,transimage):
        trans=0

        ob = math.sqrt(self.calculate_standard_derviation_single(baseimage))
        ot = math.sqrt(self.calculate_standard_derviation_single(transimage))
        obt = self.calculate_standard_derviation_AB(baseimage,transimage)
        obt=np.nan_to_num(obt)
        multiplikator=ob*ot
        multiplikator=np.nan_to_num(multiplikator)

        return obt/multiplikator

    def calculate_expecetd_value(self,array):
        return self.matrixsize * (array.sum())

    def calculate_standard_derviation_single(self,array):
        u = self.calculate_expecetd_value(array)
        #print(array)
        #print("u",u)
        derivation=np.std((array-u))
        if derivation=="nan":
            derivation=0.01

        #derivation = ((array - u).__pow__(2))
        #derivationsum=derivation.sum()
        #print("derivation",derivation)
        return self.matrixsize_plus * derivation

    def calculate_standard_derviation_AB(self,baseimage,transimage):
        uo = self.calculate_expecetd_value(baseimage)
        uc = self.calculate_expecetd_value(transimage)
        derivation = ((baseimage - uo) * (transimage- uc)).sum()
        return self.matrixsize_plus * derivation

    def get_secoundsummand(self,baseimage,transimage):
        uf = self.calculate_expecetd_value(baseimage)
        ug = self.calculate_expecetd_value(transimage)
        denominator = 2 * uf * ug
        numerator = uf.__pow__(2) + ug.__pow__(2)
        return denominator / numerator

    def get_thirdsummand(self,baseimage,transimage):
        ob2 = self.calculate_standard_derviation_single(baseimage)
        ot2 = self.calculate_standard_derviation_single(transimage)
        denominator = 2 * math.sqrt(ob2) * math.sqrt(ot2)
        numerator = ob2 + ot2
        return denominator / numerator
    def image_Quality_Index(self,baseimage,transimage):

        first = self.get_firstsummand(baseimage,transimage)
        first=np.nan_to_num(first)
        #print("first",first)
        secound = self.get_secoundsummand(baseimage,transimage)
        #print("secound",secound)
        third = self.get_thirdsummand(baseimage,transimage)
        #print("third",third)
        #print("reult Q",first * secound * third)
        return first * secound * third

    def SSIM(self,trans):

        first = self.denominator(trans)
        secound = self.numerator(trans)
        return first / secound
        pass
    def numerator(self, trans):
        ur2 = (self.calculate_expecetd_value(self.base_image_data)).__pow__(2)
        uf2 = (self.calculate_expecetd_value(trans).__pow__(2))
        c1 = 5
        firstsummand = ur2 + uf2 + c1
        derivationr = self.calculate_standard_derviation_single(self.base_image_data)
        derivationf = self.calculate_standard_derviation_single(trans)
        c2 = 5
        secoundsummand = derivationr + derivationf + c2
        return firstsummand * secoundsummand
    def denominator(self,trans):
        ur = self.calculate_expecetd_value(self.base_image_data)
        uc = self.calculate_expecetd_value(trans)
        c1 = 5
        firstsummand = 2 * ur * uc + c1
        derivationrf = self.calculate_standard_derviation_AB(self.base_image_data, trans)
        c2 = 5
        secoundsummand = 2 * derivationrf + c2
        return firstsummand * secoundsummand

