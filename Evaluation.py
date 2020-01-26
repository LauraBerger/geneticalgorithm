import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
from skimage.filters.rank import entropy
from skimage.morphology import disk
class Evaluation:
    def __init__(self):
        self.fake = cv2.imread('/Users/macbookair/Documents/Internship/results/result5.png')
        self.size = len(self.fake)
        self.matrixsize = 1.0 / (self.size * self.size)
        self.input1=cv2.imread( "/Users/macbookair/Documents/Internship/transformedImage.png",cv2.IMREAD_GRAYSCALE)
        self.input2=cv2.imread("/Users/macbookair/Documents/Internship/reference.png",cv2.IMREAD_GRAYSCALE)
        self.fused=cv2.imread("/Users/macbookair/Documents/Internship/result.png",cv2.IMREAD_GRAYSCALE)



    def standard_derivation(self):



        # Calculate the standard deviation
        # Here I'm taking the full image, you can take any rectangular region
        # Method-1: using cv2.meanStdDev()
        mean, std_1 = cv2.meanStdDev(self.fused, mask=None)

        # Method-2: using the formulae 1/n(S2 - (S1**2)/n)
        sum_1, sqsum_2 = cv2.integral2(self.fused)
        n = self.fused.size
        # sum of the region can be easily found out using the integral image as
        #  Sum = Bottom right + top left - top right - bottom left
        s1 = sum_1[-1, -1]
        s2 = sqsum_2[-1, -1]
        std_2 = np.sqrt((s2 - (s1 ** 2) / n) / n)

        print(std_1, std_2)  # [[0.45825757]] 0.4582575694


    def cross_Entropy(self):
        print("Cross Entropy Value writen own",self.cross_entropy_value())#rechnerrisch ueberpruefen
        self.object_detection()#Object detection/ Noise check:
        print ("Entropy Value iwth skimage",self.entropyvalue_with_Skimage(self.fused))
        self.texture_detection(self.fused)

    def cross_entropy(self,input):
        hist = np.histogram(input)
        histfused, binsfused = np.histogram(self.fused, bins='auto')
        histl = len(hist)
        su = 0
        # print sum(histfused)
        for p in hist:
            # print p
            for i in p:
                # print i
                r = i / sum(histfused)

                if r == 0:
                    su += 0
                else:
                    su += -r * (np.log(r))
            # print "Su",su
        return su / np.log(2)
        # print bins
    def cross_entropy_value(self):
        image1=self.input1
        cEi1 = self.cross_entropy(image1)
        #print "Inputimage 1 and fusedimage", cEi1
        cEi2 = self.cross_entropy(self.input2)
        #print "Inputimage 2 and fused image: ", cEi2
        CE = (cEi1 + cEi2) / 2
        #print "CrossEntropy: ", CE
        return CE

    def object_detection(self):
        size=len(self.fused)
        noise_mask = np.full((size, size), 28, dtype=np.uint8)
        noise_mask[32:-32, 32:-32] = 30
        img=self.fused
        noise = (noise_mask * img - 0.5 *
                 noise_mask).astype(np.uint8)
        img = noise + 128

        entr_img = entropy(img, disk(10))

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

        ax0.imshow(noise_mask, cmap='gray')
        ax0.set_xlabel("Noise mask")
        ax1.imshow(img, cmap='gray')
        ax1.set_xlabel("Noisy image")
        ax2.imshow(entr_img, cmap='viridis')
        ax2.set_xlabel("Local entropy")

        fig.tight_layout()
        plt.show()

    def entropyvalue_with_Skimage(self,img):
        return entropy(img, disk(15)).sum() / 512

    def texture_detection(self,img):
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                                       sharex=True, sharey=True)

        img0 = ax0.imshow(img, cmap=plt.cm.gray)
        ax0.set_title("Image")
        ax0.axis("off")
        fig.colorbar(img0, ax=ax0)

        img1 = ax1.imshow(entropy(img, disk(5)), cmap='gray')
        ax1.set_title("Entropy")
        ax1.axis("off")
        fig.colorbar(img1, ax=ax1)

        fig.tight_layout()

        plt.show()


    def Spatial_frequency(self):
        print ("Spatial Frequenz WITH for loop")
        print (self.Spatial_frequency_with_for())#funktioniert vom Wert her gut aber zeit ist schlecht
        print ("Spatial Frequenz EXCLUDING for loop")#fZeit gut, Wert nicht zu 100% immer richtig
        print(self.Spatial_frequency_WithOUT_for())

    def Spatial_frequency_with_for(self):
        rf2 = self.rf_for().__pow__(2)
        cf2 = self.cf_for().__pow__(2)
        return math.sqrt(rf2 + cf2)
    def cf_for(self):
        pix2 = 0
        for n in range(self.size):
            for m in range(2, self.size):
                pix = (self.fake[m, n] - self.fake[(m - 1), n]).__pow__(2)

                #            pix=(float(fake[m,n])-float(fake[(m-1),n])).__pow__(2)
                pix2 += pix
        radient = self.matrixsize * pix2
        return math.sqrt(radient.sum())
    def rf_for(self):
        pixsum = 0
        for m in range(self.size):
            for n in range(2, self.size):
                pix = (self.fake[m, n] - self.fake[m, (n - 1)])

                # pix=(float(fake[m,n])-float(fake[m,(n-1)]))
                pix2 = pix.__pow__(2)
                pixsum += pix2
        radient = self.matrixsize * pixsum
        return math.sqrt(int(radient.sum()))

    def Spatial_frequency_WithOUT_for(self):
        RF2 = self.rf().__pow__(2)
        CF2 = self.cf().__pow__(2)
        radikand = RF2 + CF2
        return math.sqrt(radikand)
    def rf(self):
        sum = (self.fake - self.fake[:, :2 - 1]).__pow__(2)
        result = sum.sum()
        radient = self.matrixsize * result
        return math.sqrt(radient)
    def cf(self):
        sum = (self.fake - self.fake[:2 - 1, :]).__pow__(2)
        result = sum.sum()
        radient = self.matrixsize * result
        return math.sqrt(radient)

