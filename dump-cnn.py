#!/usr/bin/env python

#Dumps vgg16 network activations
#By Vineet Sunkavalli

def load_model(image_size):
	import keras
	
	from keras.applications import VGG16
	vgg = VGG16(weights="imagenet", include_top=False, input_shape=image_size)#loads vgg16 convolutional head only because input shape is being modified
	
	return vgg
	
def load_images(image_size):
	import os

	cdir = os.getcwd()
	files = os.listdir(cdir)
	
	image_names = []	

	for file in files:
		if '.jpg' in file:
			image_names.append(file)

	from keras.preprocessing import image

	images = []

	for name in image_names:
		img = image.load_img(cdir+'/'+name, target_size=image_size)
		img = image.img_to_array(img)
		img /= 255#necessary to preprocess for vgg
		images.append(img)
	
	return image_names, images
	
def dump(model, image):
	import keras
	
	activations = [layer.output for layer in model.layers]
	
	from keras import models
	
	dump_model = models.Model(inputs=model.input, outputs=activations)
	dump = dump_model.predict(image)

	return dump

def save_dump(dump,image_name,sav_num):
	import cv2
	import os
	
	os.mkdir(image_name+"_dump")	
	
	import numpy as np
	
	i = 0
	for layer in dump:
		if i > sav_num:
			break

		n=0

		for channel in layer:
			if i > sav_num:
				break
			print layer.shape
			cv2.imwrite(image_name+'_dump/dmp_'+str(i)+'_'+image_name, np.uint8(255*layer[0,:,:,n]), [int(cv2.IMWRITE_JPEG_QUALITY), 40])
			i += 1
			n += 1

def main():
	names, images = load_images((150, 150))
	print "Found " + str(len(names))
	
	model = load_model((150, 150,3))

	import numpy as np
	
	for i in range(len(names)):
		print names[i]
		sav_num = raw_input("How many activations would you like to save?")
		image = np.reshape(images[i], (1,150,150,3))

		activations = dump(model, image)

		save_dump(activations, names[i], sav_num)
main()
