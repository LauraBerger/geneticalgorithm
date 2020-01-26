#abfrage nach dem file format print(image.format) # Output: JPEG
#abfrage nach pixel format print(image.mode) # Output: RGB
#abrage nach der Groesse des Bildes print(image.size) # Output: (1200, 776)
import math
from PIL import Image
import numpy as np
from numpy import linalg as LA
from PIL import Image
import time
class Fusion:

    def __init__(self, imgarr,imgarr2):
        self.size = 256

        self.pix1=imgarr
        self.pix2=imgarr2

        self.array_Pixel = np.random.randint(10, size=((self.size * self.size), 2))
        self.DataAdjust = np.zeros(((self.size * self.size), 2))

    def PCA(self):
        p=PCA()
        return p.PCA(self.pix1,self.pix2)


    def Simple_Average(self):
        for y in range(self.size):
            for x in range(self.size):
                sum = self.pix1[x, y] + self.pix2[x, y]
                PixelNew = sum / 2
                self.pix1[x, y] = int(PixelNew)

        return self.pix1

    def Simple_Maximum(self):
        for y in range(self.size):
            for x in range(self.size):
                maxi = max(self.pix1[x, y], self.pix2[x, y])
                self.pix1[x, y] = maxi
        return self.pix1

    def Simple_Minimum(self):
        for y in range(self.size):
            for x in range(self.size):
                pixel = min(self.pix1[x, y], self.pix2[x, y])
                self.pix1[x, y] = pixel
        return self.pix1

    def Weigthed_Average_Method(self,w):
        #w = 0.1
        #for x in range(10):
        for y in range(self.size):
                for x in range(self.size):
                    pixel = (w + self.pix1[x, y]) + ((1 - w) + self.pix2[x, y])
                    self.pix1[x, y] = int(pixel)
           # w += 0.1
        return self.pix1

class PCA:
    def __init__(self):
        self.size = 256



        #self.im = self.get_newtarget()
        #self.target = self.load_target()

        self.array_Pixel = np.random.randint(10, size=((self.size * self.size), 2))
        self.DataAdjust = np.zeros(((self.size * self.size), 2))

    def PCA(self,arr,arr2):


        self.array_Pixel = self.initilisar_Array_Pixel(arr)

        means = self.mean()

        adjust_matrix = (self.Assemble_the_mean_adjusted_matrix_x_and_y(means))

        matrix = self.covariance_matrix(adjust_matrix)

        w, v = LA.eig(matrix)  # eigenvalues(w) and eigenvectors(v)#evals,evecs=la.eig(matrix)

        v1 = v.T
        v = v1[np.argsort(v[1, :], axis=0)]
        p1 = self.weighted_value(v, 0)
        if p1<0:
            p1=p1*(-1)
        print('p1=',p1)
        p2 = self.weighted_value(v, 1)

        if p2<0:
            p2=p2*(-1)
        print('p2=',p2)
        self.imageFusion(p1, p2,arr,arr2)

        # im.save("/Users/macbookair/Documents/Internship/Image Datasets/gutesEndbeispiel/PCA2.png")
        eigenval = v.sort()
        return arr

    def Find_the_mean_vector(self, N, Array_Pixel):
        # N= Number of images
        # S= Dimensions: pixel of x's and pixel od y's(i.e. 100 by 100 pixels=10.000
        # S=(s1+s1+...+Sn)/n
        # aus 3*3 Bild wird 9*1 vektor
        dimensions = 0
        for y in range(5):  # forschleifen noch variable machen
            PictureFirst = 0
            for c in range(0, 1):
                OneD1 = Array_Pixel[y][c]
                PictureFirst += OneD1
                print(PictureFirst)
            dimensions = dimensions + PictureFirst
            print(dimensions)
        S = dimensions / N
        return S

    def initilisar_Array_Pixel(self, arr):
        I = np.asarray(arr)
        I.ravel()
        return self.array_Pixel
        # print Array_Pixel

    def mean(self):
        a = np.array(self.array_Pixel, dtype=int)
        return a.mean(axis=0)

    def Assemble_the_mean_adjusted_matrix_x_and_y(self, mean):
        self.DataAdjust[:, 0] = self.array_Pixel[:, 0] - mean[0]
        self.DataAdjust[:, 1] = self.array_Pixel[:, 1] - mean[1]
        return self.DataAdjust

    def calculate_Var(self, adjust_matrix, zahl):
        var = 0
        # print adjust_matrix.shape
        for x in range(10):
            # for y in range(2):
            z = pow(adjust_matrix[x][zahl], 2)
            var += z
        laenge = len(adjust_matrix)
        var_X = var / (laenge - 1)
        return var_X

    def calculate_Cov(self, adjust_matrix):
        cov = 0
        for x in range(10):
            covX = adjust_matrix[x][0]
            covY = adjust_matrix[x][1]
            covXY = covX * covY
            cov += covXY
        finalcov = cov / (10 - 1)
        return finalcov

    def covariance_matrix(self, adjust_matrix):
        varX = self.calculate_Var(adjust_matrix, 0)
        varY = self.calculate_Var(adjust_matrix, 1)
        varXY = self.calculate_Cov(adjust_matrix)
        covariance_matrix = [[varX, varXY], [varXY, varY]]
        return covariance_matrix

    def weighted_value(self, v, component):
        totalV = 0
        v1 = v[component][0]
        # print v1
        for x in range(2):
            for y in range(2):


                s = v[x][y]
                totalV += s
        # print("totalV",totalV)
        p = v1 / (totalV)
        # print ("das ist p ",p,"von",component)
        return p

    def imageFusion(self, p1, p2,arr,arr2):
        # check ob p1 und p2 positiv sind und nahe an 1 rankommen"""
        for y in range(256):
            for x in range(256):
                component1 = p1 * arr[x, y]
                component2 = p2 * arr2[x, y]
                Ifused = component1 + component2
                arr[x, y] = int(Ifused)
        return arr

    def proof_of_right_eig(self, matrix, v, w):
        u = v[:, 1]
        lam = w[1]
        eigenvec = (np.doc(matrix, u))
        eigenval = lam * u
        if eigenvec == eigenval:
            print('The Calculation ist right')








"""

def show_histogramm():#das einlesen evtl. automatisieren
    img = cv2.imread("/Users/macbookair/Documents/Internship/Image Datasets/gutesEndbeispiel/result.png")
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
target=Simple_Average(target)

#im.histogram()

"""



