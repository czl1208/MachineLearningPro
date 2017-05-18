import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
from subprocess import check_output
import matplotlib.pyplot as plt

for s in range(4):
	s=1+s
	for i in range(11):
		i=1+i
		for j in range(30):
			j=j+1
			filename = 'S'+str(s)+'/G'+str(i)+'/'+str(j)+'-depth.bin'
			ByArray = open(filename, "rb").read() 
			matrix = np.fromfile(filename,dtype=np.uint16)
			matrix = matrix.reshape(240,320)
			matrix[matrix<500] =0 
			matrix[matrix>=500] =1 
			matrix = 1-matrix
			selem = disk(2)
			matrix = binary_erosion(matrix, selem)
			matrix = label(matrix)
			areas = [r.area for r in regionprops(matrix)]
			areas.sort()
			if len(areas) > 2:
				for region in regionprops(matrix):
					if region.area < areas[-2]:
						for coordinates in region.coords:                
							matrix[coordinates[0], coordinates[1]] = 0
			matrix = matrix > 0
			crop = matrix[30:230, 50:250] 
			label_image = label(crop)
			areas = [r.area for r in regionprops(label_image)]
			areas.sort()
			if len(areas) > 1:
			    for region in regionprops(label_image):
			        if region.area < areas[-1]:
			            for coordinates in region.coords:                
			                   label_image[coordinates[0], coordinates[1]] = 0
			crop = label_image > 0


			upper=0
			while crop[upper,:].sum()==0:
				upper=upper+1


			lower=upper
			while crop[lower,:].sum()>0:

				lower=lower+1
				if(lower>=200):
					lower = 199
					break

			left=0
			while crop[:,left].sum()==0:
				left=left+1

			right=left
			while crop[:,right].sum()>0:
				right=right+1
				if(right>=200):
					right = 199
					break

			padding = np.zeros((200,200))
			pleft = int((200-(right-left))/2)
			pright = 200 - pleft
			pright = pright + (right-left) - (pright-pleft)

			pupper = int((200 - (lower-upper))/2)
			plower = 200 - pupper
			plower = plower + (lower-upper) - (plower-pupper)

			crop = crop[upper:lower+1, left:right+1]
			padding[pupper:plower+1, pleft:pright+1] = crop
			np.save('PreImage/'+str(s)+str(i)+str(j)+'.npy', padding)




























