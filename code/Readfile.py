import glob
import numpy as np
import re
# import matplotlib.pyplot as plt
import csv

class DataSet(object):
	"""docstring for data"""
	def __init__(self, train_data, train_label, test_data, test_label):
		self.train_data = train_data
		self.train_label = train_label
		self.test_data = test_data
		self.test_label = test_label
		
		
def readfile(Dpath, labelFile):
	with open(labelFile, mode='r') as infile:
		reader = csv.reader(infile)
		for rows in reader:
			label_dict = {rows[0]:rows[1] for rows in reader}

	path = Dpath+'/*.npy'   
	files=glob.glob(path) 
	row = len(files)
	data_set = np.zeros((row, 40000))
	label_set = np.zeros((row, 1)) 
	r=0 
	for file in files:
		labelIndex=re.findall(r'\d+', file)
		iD=labelIndex[0]
		label = int(label_dict.get(iD))
		label_set[r] = label   
		im = np.load(file)
		vector = im.reshape(1,40000)
		data_set[r,:] = vector
		r=r+1
	test_label = np.zeros((110, 1))
	test_data = np.zeros((110, 40000))
	train_label = np.zeros((row-110, 1))
	train_data = np.zeros((row-110, 40000))
	i=0;
	while i<110:
		test_label[i] = label_set[10*i]
		test_data[i,:] = data_set[10*i,:]
		train_data[9*i:9*(i+1),:] = data_set[10*i+1 : 10*(i+1), :]
		train_label[9*i:9*(i+1)] = label_set[10*i+1 : 10*(i+1)]
		i=i+1
	train_data[9*i:row,:] = data_set[10*i : row, :]
	train_label[9*i:row] = label_set[10*i : row]
	data_set = DataSet(train_data, train_label, test_data, test_label)
	return data_set

data_set = readfile('PreImage', 'label.csv')



