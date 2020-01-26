from PIL import Image
import pywt
from Image_FusionSimple import PCA,Fusion
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
class DWT:
    def __init__(self,image1,image2):
        self.titels = ['Approximation', ' Horizontal detailLv.2',
                       'Vertical detailLv.2', 'Diagonal detailLv.2', ' Horizontal detail',
                       'Vertical detail', 'Diagonal detail']
        self.image1=image1
        self.image2=image2
        self.size = 256

        self.im = self.get_newtarget()
        self.target = self.load_target()




        self.array_Pixel = np.random.randint(10, size=((self.size * self.size), 2))
        self.DataAdjust = np.zeros(((self.size * self.size), 2))
        pass

    def dwt(self):
        #imLists = [IMAGEPATH + "a01_1.tif", IMAGEPATH + "a01_2.tif"]
        start=time.time()
        self.show_Image_DWt(self.image1)
        self.show_Image_DWt(self.image2)

        # show_DWt1D()
        coeffs = self.calculate_coeffs(self.image1)
        arr, coeff_slices = pywt.coeffs_to_array(coeffs)
        #show_DWt1D(coeffs)
        # show_Image_DWt(coeffs,target)
        #self.show_DWt1D()

        coeffs2 = self.calculate_coeffs(self.image2)
        arr2, coeff_slices2 = pywt.coeffs_to_array(coeffs2)
        startPCA=time.time()
       # p=PCA()
        #array = p.PCA(arr, arr2)#FusionMethod
        f=Fusion(arr,arr2)
        array=f.PCA()
        endPCA=time.time()
        print('Gesamtzeit PCA: {:5.3f}s'.format(endPCA - startPCA))
        coeffs_from_arr = pywt.array_to_coeffs(array, coeff_slices, output_format='wavedecn')
        #self.show_DWt1D()
        self.target = pywt.waverecn(coeffs_from_arr, wavelet='db1')
        self.plot()
        end=time.time()
        print('Gesamtzeit DWT: {:5.3f}s'.format((end - start)-(endPCA-startPCA)))
        print('Gesamtzeit DWT and PCA: {:5.3f}s'.format(end - start))
        return self.target


    def get_newtarget(self):
        im = Image.new("L", (self.size, self.size))
        return im
    def load_target(self):
        return self.im.load()
    def show_target(self):
        self.im.show()
    def save_target(self):
        self.im.save("/Users/macbookair/Documents/Internship/Image Datasets/gutesEndbeispiel/resultGeneticAlgorithm.png")

    def calculate_coeffs(self,image):
        return pywt.wavedecn(image, 'db1', level=2)
    def imageloader(self,path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to float for more resolution for use with pywt
        image = np.float32(image)
        # image /= 255
        return image

    def show_Image_DWt(self,img):
        coeffs = pywt.dwt2(img, 'bior1.3')
        LL, (LH, HL, HH) = coeffs
        coeffs2 = pywt.dwt2(LL, 'bior1.3')
        L2, (LH2, HL2, HH2) = coeffs2
        for y in range(len(L2)):
            for x in range(len(L2)):
                sum1 = L2[x, y]
                self.target[x, y] = int(sum1)
        for y in range((len(LH2))):
            for x in range(len(LH2)):
                pix = int(LH2[x, y])
                self.target[(x + 64), y] = int(pix)
        for y in range(len(HL2)):
            for x in range(len(HL2)):
                self.target[x, (y + 64)] = int(HL2[x, y])
        for y in range(len(HH2)):
            for x in range(len(HH2)):
                self.target[(x + 64), (y + 64)] = int(HH2[x, y])
        for y in range(len(LH)):
            for x in range(len(LH) - 2):
                pix = int(LH[x, y])
                self.target[(64 * 2 + x), (y)] = pix
        for y in range((len(HL) - 2)):
            for x in range(len(HL)):
                self.target[(x), (y + 128)] = int(HL[x, y])
        for y in range(len(HH) - 2):
            for x in range(len(HH) - 2):
                # target=[(x+(64*64)),(y+64*64)]=sum7
                self.target[(x + (64 * 2)), (y + 128)] = int(HH[x, y])
        self.im.show()
    def show_DWt1D(self):
        coeffs2 = pywt.dwt2(self.image1, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        fig = plt.figure(figsize=(12, 3))
        for i, a in enumerate([LL, LH, HL, HH]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_title( self.titels[i], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()
        self.DWt1Dto2D(LL)
    def DWt1Dto2D(self,LL):
        coeffs2 = pywt.dwt2(LL, 'bior1.3')
        L2, (LH2, HL2, HH2) = coeffs2
        #fig = plt.figure(figsize=(12, 3))
        plt.imshow(np.hstack((L2,LH2,HL2,HH2)), cmap="gray")
        #for i, a in enumerate([L2, LH2, HL2, HH2]):
         #   ax = fig.add_subplot(1, 4, i + 1)
         #   ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
          #  ax.set_title( self.titels[i], fontsize=10)
           # ax.set_xticks([])
            #ax.set_yticks([])

        #fig.tight_layout()
        plt.show()
    def plot(self):
        plt.figure(0)
        plt.gray()
        plt.subplot(131)
        plt.imshow(self.image1)
        plt.subplot(132)
        plt.imshow(self.image2)
        plt.subplot(133)
        plt.imshow(self.target)
        plt.show()
        plt.plot(self.target)
        plt.show()