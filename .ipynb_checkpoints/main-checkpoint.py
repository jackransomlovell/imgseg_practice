from skimage.io import imread_collection, imshow_collection
from skimage.filters import threshold_multiotsu
from skimage.color import rgb2gray
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.filters import median
from skimage import feature
from skimage.color import label2rgb
from scipy import ndimage as ndi
from matplotlib.colors import ListedColormap
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Img. Seg. and Instance Seg. tutorial')
    parser.add_argument('--img_path',
                        default ='data/NucleusSegDataExtended/TrainingSet/*.jpg',
                        help='Path to imgs')
    args = parser.parse_args()
    img_path = args.img_path
    col = imread_collection(img_path)
    #convert to np array
    np_col = np.array(col)
    #original fig
    og_fig = imshow_collection(col)
    og_fig.savefig('og_fig')
    #conver to grayscale
    gray_col = rgb2gray(col)
    #save grayscale
    gray_fig = imshow_collection(gray_col)
    gray_fig.savefig('gray_fig', cmap = 'gray')
    #get avg otsu's thrsh on collection
    avg_thr = threshold_multiotsu(gray_col)
    #apply threshold to images
    avg_thr_imgs = np.digitize(gray_col, bins=avg_thr)
    #make a dict of each indiv thresh, would like to figure out how to vectorize if possible
    indiv_thr = {}
    for key, val in enumerate(gray_col):
        indiv_thr[key] = threshold_multiotsu(val)
    #make new dict for indiv thresholded imgs
    indiv_thr_imgs = [np.digitize(img, bins=thr) for thr, img in 
                          zip(indiv_thr.values(), gray_col)]
    #turn thrshd vals into a numpy arrays so we can compute the diff
    np_indiv_thr = np.array(list(indiv_thr_imgs.values()))
    np_avg_thr = np.array(avg_thr_imgs)
    #get difference b/w each
    diff_imgs = avg_thr_imgs ^ np_indiv_thr
    
    #save each image collection
    avg_fig = imshow_collection(avg_thr_imgs,cmap = 'gray')
    avg_fig.savefig('avg_thr_fig')
    indiv_fig = imshow_collection(np_indiv_thr,cmap = 'gray')
    indiv_fig.savefig('indiv_thr_fig')
    diff_fig = imshow_collection(diff_imgs,cmap = 'gray')
    diff_fig.savefig('diff_fig')
    #now compute instance seg.
    #get eucl. dist. of mask from bkgd for avg
    avg_thr_distance = ndi.distance_transform_edt(np_avg_thr)
    #and indiv
    indiv_thr_distance = ndi.distance_transform_edt(np_indiv_thr)
    
    #now compute watershed w/ ndi 
    #first compute local max, to tell us where the max distances of our nuclei are 
    #from the bground
    avg_local_maxima = ndi.maximum_filter(avg_thr_distance, size=21)
    indiv_local_maxima = ndi.maximum_filter(indiv_thr_distance,size=21)
    #next label the connected regions
    avg_markers, _ = ndi.label(avg_local_maxima)
    indiv_markers, _ = ndi.label(indiv_local_maxima)
    #now compute watershed
    avg_labels = watershed(-avg_thr_distance, avg_markers, mask=np_avg_thr)
    indiv_labels = watershed(-indiv_thr_distance, indiv_markers, mask=np_indiv_thr)
    
    # Use ListedColormap to create a random colormap to
    # help visualize the results
    cmap = ListedColormap(np.random.rand(256,3))
    avg_fig = imshow_collection(avg_labels)
    avg_fig.savefig('avg_inst_fig', cmap=cmap)
    indiv_fig = imshow_collection(indiv_labels)
    indiv_fig.savefig('indiv_inst_fig', cmap=cmap)
    
    ## now compute contours of each
    
    
    
if __name__ == '__main__':
    main()