#!/usr/bin/env python
import cStringIO
import numpy as np
from PIL import Image
import urllib2
import cv2
import os
import random
import time
import multiprocessing as mp
import time
from os import error
import pickle



path = os.getcwd()
reverb = pickle.load(open('/users/mkrzus/reverbtop20.p'))

#dictionary data {hashkey: x:link, y:class}
#{'eecf86c7bd6f110633261693b2e980e7':
#  {'x': [u'https://reverb-res.cloudinary.com/image/upload/a_exif,c_limit,fl_progressive,h_1136,q_85,w_640/v1425061035/nos4ed2g3cpdcltlrtwm.jpg'],
#   'y': u'epiphone'}}

#for getting the pictures
userAgents = [
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.67 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; rv:31.0) Gecko/20100101 Firefox/31.0',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:25.0) Gecko/20100101 Firefox/25.0',
    'Mozilla/5.0 (X11; CrOS i686 3912.101.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.116 Safari/537.36',
    'Mozilla/4.61 [ja] (X11; I; Linux 2.6.13-33cmc1 i686)',
    'Opera/9.63 (X11; Linux x86_64; U; ru) Presto/2.1.1',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/534.55.3 (KHTML, like Gecko) Version/5.1.3 Safari/534.53.10'
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.142 Safari/535.19',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.7; rv:11.0) Gecko/20100101 Firefox/11.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:8.0.1) Gecko/20100101 Firefox/8.0.1',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.151 Safari/535.19',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20100121 Firefox/3.5.6 Wyzo/3.5.6.1'
]

def safetyMeasures(wait_time):
    version = random.choice(userAgents)
    header = { 'User-Agent' : version }
    time.sleep(random.randint(0,wait_time))
    return header

def pageGetter(website,time):
    header = safetyMeasures(time)
    req = urllib2.Request(website,headers = header)
    soup = urllib2.urlopen(req).read()
    return soup


def transform_size(d,size=(350,350)):
	file = cStringIO.StringIO(pageGetter(d,random.randint(0,0)))
	img = Image.open(file)
	img.thumbnail(size, Image.ANTIALIAS)
	if img.size[1] == img.size[0]:
		img = np.array(img)
	#img = cv2.copyMakeBorder(img,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])
	#either (500,>)
	#either (<,500)
	elif img.size[1]<size[0]: #x shape[0]
		img = np.array(img)
		val = abs(size[0] - img.shape[0])
		new_val = val/2
		if new_val*2 < val:
			add = abs(val-new_val)
			img = cv2.copyMakeBorder(img,new_val,add,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
		else:
			img = cv2.copyMakeBorder(img,new_val,new_val,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])

	elif img.size[0]<size[0]: #x shape[0]
		img = np.array(img)
		val = abs(size[0] - img.shape[1])
		new_val = val/2
		if new_val*2 < val:
			add = abs(val-new_val)
			img = cv2.copyMakeBorder(img,0,0,new_val,add,cv2.BORDER_CONSTANT,value=[255,255,255])
		else:
			img = cv2.copyMakeBorder(img,0,0,new_val,new_val,cv2.BORDER_CONSTANT,value=[255,255,255])
	return img


class Transformer(object):
	def __init__(self, folder_path=path):
		self.PATH_CREATED = False
		self.path = folder_path

	def transform(self, reverbdic):
		try:
			k,v = reverbdic
			os.chdir(self.path)
			if self.PATH_CREATED == False:
				try:
					os.listdir(self.path+'/images')
				except error:
					os.mkdir(self.path+'/images')
					self.PATH_CREATED = True
					os.chdir(self.path+'/images')

			if k not in os.listdir(self.path+'/images'):
				os.chdir(self.path+'/images')
				print(k)
				os.mkdir(k)
				os.chdir(k)
				label = v['y']
				img = Image.fromarray(transform_size(v['x'][0]))
				img.save(k+'.png')
				with open('label.txt','wb') as f:
					f.write(label)
				os.chdir(self.path+'/images')
		except:
			pass

if __name__ == '__main__':
	trans = Transformer()
	for k,v in reverb.items():
