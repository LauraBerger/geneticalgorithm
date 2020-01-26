from PIL import ImageEnhance

from FitnessCalculation import  Histogram
from gai import  GA
from PIL import Image as IM
import Image_FusionSimple as fm
from Image_FusionSimple import Fusion
import math
import numpy as np
import matplotlib.pyplot as plt
from Image import Image
import cv2
from DWT import DWT
from Evaluation import  Evaluation

from skimage import  io
import time
# kontrast verbessern
def set_images():
    PathDicom = "./images/test/"
    #PathDicom = "/Users/macbookair/Documents/Internship/Image Datasets/test/data"

    image1 = Image(PathDicom, True, 0)
    image1.resize_image(256)
    image2 = Image(PathDicom, True, 1)
    image2.resize_image(256)
    image1.transform_image_par(trans=(0.0, 0.0), scale=(1, 1), rotation=0.0, shear=(0.0, 0.0))
    image2.transform_image_par(trans=(0, 0), scale=(1, 1), rotation=0, shear=(0, 0))
    #plt.imshow(np.hstack((image1.get_image(), image2.get_image())), cmap="gray")  # plt.cm.binary/bone
   # plt.show()
    # image1.histogrammstauchung()
    # enhancer = ImageEnhance.Contrast(image1)
    # enhanced_im = enhancer.enhance(4.0)
    # enhanced_im.save("enhanced.sample1.png")
    return image1,image2


def set_histogramms(image1,image2):
    h = Histogram()
    # h.show_histogram(image1.get_image())
    # h.show_histogram(image2.get_image())
    # h.show_joint_histogram(image1.get_image(),image2.get_image(),log=True)

    mi = h.mutual_information(h.calc_joint_histogram(image_1=image1.get_image(), image_2=image2.get_image()))
    # print(mi)
    #   nmi =h.normalized_mutal_information(h.calc_joint_histogram(image1.get_image(),image2.get_image()))
    # print(nmi)
    best_mi = h.mutual_information(h.calc_joint_histogram(image_1=image1.get_image(), image_2=image1.get_image()))
    print(best_mi)

def genetic_algorithm(image1,image2):
    ga = GA(chromosomes_equal=True, population_size=32, base_image=image1, trans_image=image2)
    ga.genetic_algorithm_cycle()
    fittest = ga.best_solution()
    # print("\n" + "\033[92m" + "fittest individual:" + str(fittest[0])+ " Mi: " + str(fittest[1])+ "\033[0m")

    ga_solution = image2.calc_transform_image(ga.best_solution()[0])
    # print(fittest)
    plt.imshow(np.hstack((image1.get_image(), ga_solution)), cmap="gray")
    plt.show()
    # f=fit(image1,image2)
    # Q=f.image_Quality_Index(image1.get_image(),image2.get_image())
    # print("Image Quälity Index",Q)
    return ga_solution

def fusion(image1,ga_solution):
    # t=Fusion(filepath_1)
    # target=Fusion.Simple_Average(t)
    # t.show_target()
    # t.save_target()
    dw = DWT(image1.get_image(), ga_solution)
    target = DWT.dwt(dw)
    return target

def evaluation():
    e = Evaluation()
    e.standard_derivation()
    e.Spatial_frequency()
    e.cross_Entropy()

def save_all_3_images(image1, ga_solution, target):
    outdir = './'
    cv2.imwrite("/Users/macbookair/Documents/Internship/transformedImage.png", ga_solution)
    cv2.imwrite("/Users/macbookair/Documents/Internship/reference.png", image1.get_image())

    cv2.imwrite("/Users/macbookair/Documents/Internship/result.png", target)

def main():
    image1,image2=set_images()
    set_histogramms(image1,image2)
    ga_solution=genetic_algorithm(image1,image2)
    target=fusion(image1,ga_solution)

    save_all_3_images(image1,ga_solution,target)
    evaluation()

if __name__ == "__main__":
    main()




