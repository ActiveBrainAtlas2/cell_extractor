import pickle as pkl
import cv2
import numpy as np

# with open('mean.pkl','br') as pkl_file:
#     stat=pkl.load(pkl_file) 

# mean_s=stat['Mean']

def sobel(img):
    """ Compute the normalized sobel edge magnitudes """
    sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    _mean=(np.mean(sobel_x)+np.mean(sobel_y))/2.
    _std=np.sqrt((np.var(sobel_x)+np.var(sobel_y))/2)
    sobel_x=(sobel_x - _mean)/_std
    sobel_y=(sobel_y - _mean)/_std
    return sobel_x, sobel_y

def trim_array_to_size(array,size0,size2):
    if(array.shape[0]>size0):
        size_difference=int((array.shape[0]-size0)/2)
        array=array[size_difference:size_difference+size0,:]
    if(array.shape[1]>size2):
        size_difference=int((array.shape[1]-size2)/2)
        array=array[:,size_difference:size_difference+size2]
    return array

def equalize_array_size_by_trimming(array1,array2):
    size0=min(array1.shape[0],array2.shape[0])
    size1=min(array1.shape[1],array2.shape[1])
    array1=trim_array_to_size(array1,size0,size1)
    array2=trim_array_to_size(array2,size0,size1)
    return array1,array2

def calc_img_features(img,mean_s):
    """  
         img = input image
         mean_s: the untrimmed mean image
         Computes the agreement between the gradient of the mean image and the gradient of this example
         mean_x,mean_y = the gradients of the particular image
         img_x,img_y = the gradients of the image

    """
         
    img,mean=equalize_array_size_by_trimming(img,mean_s)
    mean_x,mean_y=sobel(mean)
    img_x,img_y=sobel(img)
    
    dot_prod = (mean_x*img_x)+(mean_y*img_y)
    corr=np.mean(dot_prod.flatten())      #corr = the mean correlation between the dot products at each pixel location
    
    mag=np.sqrt(img_x*img_x + img_y*img_y)
    energy=np.mean((mag*mean).flatten())  #energy: the mean of the norm of the image gradients at each pixel location
    return corr,energy
