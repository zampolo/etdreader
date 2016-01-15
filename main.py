## @file main.py
# **main.py:**  *Test file for the etdreader module*
#
# Author: Ronaldo de Freitas Zampolo 
#
# Begining: 07.jan.2016 
#
# Version 1.0: 14.jan.2016 
# 
#
#
# 
#
#
import matplotlib.pyplot as plt
import numpy as np

print(
'''
===========================================================
  Welcome to this test file for the 'etdreadre.py' module
===========================================================

Author: Ronaldo de Freitas Zampolo
Version 1.0 (14.jan.2016)

This provides a step-by-step in order to follow the execution.

''' )

print('- Loading the module')
# loading the etdreader module
import etdreader as etdr

# @var path_images
# Path where data files are located
path_images = '/home/zampolo/engenharia/projetos/posdoc/data/ivc/eyetrack_live/images/'

# @var image_name
# test image adopted for showing the visual output (all data are analised but only Saliency Map for this one is exhibited)
image_name = 'paintedhouse' #'lighthouse2'

# @var img
# Test image
# Reading test image
print('- Reading the test image')
img = plt.imread(path_images+image_name+'.bmp')

# @var files
# Such a variable is a list, whose membres store the whole name of a data file (path, name and extension).
print('- Discovering the names of data files inside the test data repository')
files = etdr.find_files() # this function is called with default parameters

# @var ns
# List that stores the number of samples of each test
# @var ca
# List that stores the calibrations area of each test (in pixels)
# @var hd
# List that stores the head distance of each test (in mm)
# @var sp
# List that stores the sampling rate of each test
# @var ti
# List that stores the time vector of each test. Such a time vector contains for
# each point of regard sample the time label recorded
# @var tp
# List that stores the type vector of each test. Such a time vector contains for
# each point of regard sample the type label recorded (SMP, sample; MSM, message)
# @var lporx
# List that stores the x-vector of the left POR of each test.
# @var lpory
# List that stores the y-vector of the left POR of each test.
# @var rporx
# List that stores the x-vector of the right POR of each test.
# @var rpory
# List that stores the y-vector of the right POR of each test.

# Retrieving all information from data files from a folder
print('- Retrieving some information from data files (it may take some time)')
ns, ca, hd, sp, ti, tp, lporx,lpory,rporx,rpory = etdr.extract_information(files) 

# @var imageinfo
# List with
# Reading resolution and calculating vertical and horizontal shifts of test images,
# based on their sizes and the size of the screen
print('- Loading information from test files')
imageinfo = etdr.load_image_information(screen_size=ca[0])

# @var gp
# Dictionary that group data by image
# gp = {image_name: [ leftPorX, leftPorY, rightPorX, rightPorY ] } (binocular data)
print('- Grouping POR data by test image (it may take some time)' )
gp = etdr.grouping_por_by_image( files, lporx, lpory, rporx, rpory)

# @var fg
# Dictionary that group fixations by image
# fp = {image_name: [ subject 1, subject 2, ...] } (dictionary of lists)
# where:
#   subject = [ fixation1, fixation2, ...]  (list)
#   fixation = {x: fixation x-vextor, y: fixation y-vector, xAvg: x-vector average, yAvg: y-vector average,
#               timeBegin: fixation initial time, timeFinal: fixation final time}

# Calculate fixation ponts from binocular PORs and corresponding time vector
print('- Calculating fixations (it may take A LOT OF time)')
print('As some warnings may appear, hints are given to show that everything is going fine:')
fg = etdr.fixation_detection_and_grouping(files,lporx,lpory,rporx,rpory,ti)

print('- Generating visual output')
# Generating visual outputs
# As data is binocular, we first calculate the average point of regard
# Original POR data
x = (gp[image_name][0]+gp[image_name][2])/2
y = (gp[image_name][1]+gp[image_name][3])/2

# @var porMat1
# POR matrix: it contains all the POR points for a given test image
# @var porImg
# Test image as well as all retrieved PORs

# Generating output images/matrices
porMat1, porImg = etdr.mapping_por_into_image( image = img, image_por= [x,y], shift=imageinfo[image_name][1], colour = [0,0,255])

# Adding fixation data to the previous image
x=[]
y=[]
for each_subject in fg[image_name]:
    for each_fixation in each_subject:
        x.append(each_fixation['xAvg'])
        y.append(each_fixation['yAvg'])
x = np.array(x)
y = np.array(y)

# @var fixMat
# Fixation matrix: it contains all the fixation points for a given test image
# @var porImg
# Additing the fixation points to the previous version

# Generating output images/matrices
fixMat, porImg = etdr.mapping_por_into_image( image = porImg, image_por= [x,y], shift=imageinfo[image_name][1], colour = [0,255,0])

# @var sMap
# Saliency map calculated from the Fixation Matrix

# Generating the saliency map
print('- Calculating the saliency map for the example (', image_name, ')')
sMap = etdr.saliency_map_from_fixations( fixMat )

# --- Print-out of some retrieved information --- #
print('----------- Print-outs -----------------------')
print('Number of samples of the first file data read: ',ns[0])
print('Calibration area of the first file data read: ',ca[0],' pixels')
print('Head distance of the first file data read: ',hd[0], ' mm')
print('Sampling rate of the first file data read: ',sp[0], ' Hz')
print('Number of time points of the first file data read: ',len(ti[0]))

print('---')
print('Number of test images: ',len(fg))
print('Number of subjects for image ', image_name,': ',len(fg[image_name]))
print('Number of fixations of subject 0: ', len(fg[image_name][0]))
print(image_name,'- resolution: ',imageinfo[image_name][0],' shift: ',imageinfo[image_name][1])

# --------------------------------------------------------------------- #

# --- Visual output --- #
print('---')
print('Exhibiting images')
# Test image
plt.imshow(img)
plt.title('Original image: '+ image_name )

# Test image with PORs and Fixation points
#(the latter are not much apparent, but they are there)
plt.figure()
plt.imshow(porImg)
plt.title(image_name+': POR (blue) and fixations (other color) on image')

# Saliency map
plt.figure()
plt.imshow(sMap)
plt.title(image_name+': saliency map')
plt.show()
# --------------------------------------------------------------------- #
