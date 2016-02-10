## @file main.py
#
# @brief Test file for the etdreader module
# @detailes
# @author Ronaldo de Freitas Zampolo 
# @version 1.0
# @date 14.jan.2016  
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

# @var path_images path where data files are located
path_images = '/home/zampolo/engenharia/projetos/posdoc/data/ivc/eyetrack_live/images/'

# @var image_name test image name
# @brief adopted for showing the visual output (all data are analised but only Saliency Map for this one is exhibited)
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
fixMat, porImg2 = etdr.mapping_por_into_image( image = img, image_por= [x,y], shift=imageinfo[image_name][1], colour = [0,0, 255])

# Generating scanpaths
spImage = etdr.plot_scanpath(image = img, image_fix = [x[0:9],y[0:9]], shift=imageinfo[image_name][1], r=10) 

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

# Saving processed images
## PNG versions
plt.imsave(fname = 'fixations.png',arr=porImg2)
plt.imsave(fname = 'scanpaths.png', arr=spImage)
plt.imsave(fname = 'fixmap.png', arr=sMap, cmap = 'gray')
plt.imsave(fname = 'fixmapC.png', arr=sMap)
## EPS versions
#plt.imsave(fname = 'fixations.eps',arr=porImg2)
#plt.imsave(fname = 'scanpaths.eps', arr=spImage)
#plt.imsave(fname = 'fixmap.eps', arr=sMap, cmap = 'gray')
#plt.imsave(fname = 'fixmapC.eps', arr=sMap)
#-------------------------------------------------------------

# Generating supperposition (find another way to do without the need to save a png first)
fixMC = plt.imread('fixmapC.png') # reading PNG 
fixMC = fixMC[:,:,0:3] # taking the A channel away
fix_img = etdr.hotmap(img/255.0, fixMC) # normalising image and calling the superposition function
plt.imsave(fname = 'fixImg.png', arr=fix_img) # salving the result (PNG)
#plt.imsave(fname = 'fixImg.eps', arr=fix_img) # salving the result (EPS)

# --- Visual output --- #
print('---')
print('Exhibiting images')
# Test image
plt.imshow(img)
plt.title('Original image: '+ image_name )

# Test image with PORs and Fixation points
#(the latter are not much apparent, but they are there)
plt.figure()
plt.imshow(porImg2)
plt.title(image_name+': POR (blue) and fixations (other color) on image')

           
# Test image with some scanpaths
plt.figure()
plt.imshow(spImage)
plt.title(image_name+': Some scanpath')
           
# Fixation map
plt.figure()
plt.imshow(sMap, cmap = 'gray')
plt.title(image_name+': saliency map')

# Hotmap
plt.figure()
plt.imshow(fix_img)
plt.title(image_name+': hotmap')

# Example of POR (distinction between saccades and fixations)
por = np.array( lporx[0] )
por = por[0:3999]

plt.figure()
plt.grid(True)
plt.plot(por, 'b')
#plt.title(image_name+': example of POR')
plt.xlabel('amostras')
plt.ylabel('posição do olhar (x)')
#plt.text(500,700,'fixação 1')
plt.annotate('fixação 1',xy=(750,650),xytext=(500,700),arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.text(2500,950,'fixação 2')
plt.annotate('fixação 2',xy=(2750,900),xytext=(2500,950),arrowprops=dict(facecolor='black', shrink=0.05),)
plt.axis([0,3999,500,1000])
plt.savefig('por.pdf', dpi = 1200) # saving in PDF
plt.savefig('por.eps', dpi = 1200) # saving in EPS

plt.figure()
plt.grid(True)
plt.plot(por, 'b')
#plt.title(image_name+': example of POR')
plt.xlabel('amostras')
plt.ylabel('posição do olhar (x)')
plt.axis([2150,2300,500,1000])
plt.annotate('movimento sacádico',xy=(2215,750),xytext=(2240,745),arrowprops=dict(facecolor='black', shrink=0.05),)
plt.savefig('porDet.pdf', dpi = 1200) # saving in PDF
plt.savefig('porDet.eps', dpi = 1200) # saving in EPS

plt.show()
# --------------------------------------------------------------------- #
