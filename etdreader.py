## @namespace etdreader
#
# @brief Read eye tracking data files in IVC format
#
# @author Ronaldo de Freitas Zampolo
# @version 1.0
# @date 14.jan.2016
# @date 07.jan.2016 (begining)
#

import os
import numpy as np
import scipy.signal as sig
import scipy.ndimage.filters as filt2D
#import cv2 

## 
# @brief This function retrieves all file names in *path* with any of given extentions in *type*
#
# @param path data file path *string*
# @param types data file extention *list of strings*
#
# @retval found_files *list of strings* containing all the files of a given *path* with extension *type*
# 
def find_files( path = '/home/zampolo/engenharia/projetos/posdoc/data/ivc/eyetrack_live/etdata/', types = ['.txt'] ):
    try:
        found_files = os.listdir(path)
        for each_file in found_files:
            file_type = os.path.splitext(each_file)
            if (not(file_type[1] in types)):
                found_files.remove(each_file)
        
        return sorted(found_files)
    except:
        print('Some problem has occurred. Verify input parameters.')
#-----------------

## 
# @brief Extract relevant information from data files and store it into variables for further processing
# @param list_of_file_names list with all test file names (*list of strings*)
# @param  path where all data files are (*string*)
#
# @retval  number_of_samples  list with the number of samples for each test (*list of string*).
# @retval  calibration_area  list with the screen calibration area for each test (*list of tuple of integers*). Each element is a *tuple of integers* (X,Y), where X and Y correspond to the horizontal and vertical resolutions respectively.
# @retval  head_distance list with the distance between subject head and screen in mm for each test (*list of string*)
# @retval  sampling_rate list with the nominal sampling rate for each test in Hz (*list of string*)
# @retval  time_matrix list of time vector for each test. The time vector contains the value of system clock for every sampling time in micro seconds (list of floats)
# @retval  type_matrix  list of data type (MSG: message or SMP: sample) vectors for each test. Each data type vector contains a string that indicates the type of data for every sample (*list of string*)
# @retval  lporx_matrix matrix with the left point of regard (POR) x vector for each test (*list of floats*)
# @retval  lpory_matrix matrix with the left point of regard (POR) y vector for each test (*list of floats*)
# @retval  rporx_matrix matrix with the right point of regard (POR) x vector for each test (*list of floats*)
# @retval  rpory_matrix matrix with the right point of regard (POR) y vector for each test (*list of floats*)
#
# @attention Attention to output types
#
def extract_information(list_of_file_names, path = '/home/zampolo/engenharia/projetos/posdoc/data/ivc/eyetrack_live/etdata/'):
    # output lists initialisation
    number_of_samples = []
    calibration_area = []
    head_distance = []
    sampling_rate = []
    time_matrix = []
    rporx_matrix = []
    rpory_matrix = []
    lporx_matrix = []
    lpory_matrix = []
    type_matrix = []
    
    # list of files loop
    for each_file in list_of_file_names:
        # opening each data file
        data_file = open(path+each_file,'r')

        # initialising/reseting temporarily data lists
        time_vector = []
        rporx_vector = []
        rpory_vector = []
        lporx_vector = []
        lpory_vector = []
        type_vector = []
        
        # file loop
        for each_line in data_file: # reading each line of file data
            
            # spliting line (separation: general blank space)
            splited_line = each_line.split(sep = None)

            # identifying the type of information contained in each line
            if '##' in splited_line:

                # number of samples
                if 'Number' in splited_line:
                    number_of_samples.append(splited_line[4])

                # calibration area (screen resolution)
                elif 'Area:' in splited_line:
                    calibration_area.append( (int(splited_line[3]),int(splited_line[4])))

                # Viewing distance
                elif 'Head' in splited_line:
                    head_distance.append(splited_line[4])

                # Sampling rate
                elif 'Sample' in splited_line:
                    sampling_rate.append(splited_line[3])

            # data portion of the file
            elif ( not('Time' in splited_line) &  len(splited_line) > 0 ):
                # time information
                time_vector.append(float(splited_line[0]))
                # type of data (sample or message)
                type_vector.append(splited_line[1])
                # POR
                lporx_vector.append(float(splited_line[17])-1)
                lpory_vector.append(float(splited_line[18])-1)
                rporx_vector.append(float(splited_line[19])-1)
                rpory_vector.append(float(splited_line[20])-1)
                
                #tv_vector.append(splited_line[21])
                
        # output update
        time_matrix.append(time_vector)
        type_matrix.append(type_vector)
        lporx_matrix.append(lporx_vector)
        lpory_matrix.append(lpory_vector)
        rporx_matrix.append(rporx_vector)
        rpory_matrix.append(rpory_vector)
        
        # closing data file
        data_file.close()
    
    return number_of_samples, calibration_area, head_distance, sampling_rate, time_matrix, type_matrix, lporx_matrix, lpory_matrix, rporx_matrix, rpory_matrix 

##
# @brief Read test images' resolutions and calculate offsets.
# @detailes The offset are horizontal and vertical shifts that are calculated from image resolution (in pixels) and from screen size (in pixels too). Such an offset is necessary to plot correctly the PORs, which are given with respect to screen coordinates, in the corresponding image matrices. In other words, it is just a translation in the reference sytem. It is assumed the test images were exhibited in the centre of the screen during tests.
# 
# @param screen_size dimensions of the screen (*tuple of integers*)
# @param name name of data base of interest (*string*)
#
# @retval imginfo *dictionary of list* whose keys are the names of test images. The list is composed of two elements: the first one is the resolution of the test image (*tuple of integers*) and the second is the calculated offsets (*tuple of floats*)
#
def load_image_information(screen_size,name='live'):
    if name == 'live':
        imginfo = {'bikes': [(768,512)],
                   'building2':[(640,512)],
                   'buildings':[(768,512)],
                   'caps':[(768,512)],
                   'carnivaldolls':[(610,488)],
                   'cemetry':[(627,482)],
                   'churchandcapitol':[(634,505)],
                   'coinsinfountain':[(614,512)],
                   'dancers':[(618,453)],
                   'flowersonih35':[(640,512)],
                   'house':[(768,512)],
                   'lighthouse':[(480,720)],
                   'lighthouse2':[(768,512)],
                   'manfishing':[(634,438)],
                   'monarch':[(768,512)],
                   'ocean':[(768,512)],
                   'paintedhouse':[(768,512)],
                   'parrots':[(768,512)],
                   'plane':[(768,512)],
                   'rapids':[(768,512)],
                   'sailing1':[(768,512)],
                   'sailing2':[(480,720)],
                   'sailing3':[(480,720)],
                   'sailing4':[(768,512)],
                   'statue':[(480,720)],
                   'stream':[(768,512)],
                   'studentsculpture':[(632,505)],
                   'woman':[(480,720)],
                   'womanhat':[(480,720)]}
    # Detemining offsets
    for img in imginfo:
        shiftx = (screen_size[0] - imginfo[img][0][0])//2 #+1
        shifty = (screen_size[1] - imginfo[img][0][1])//2 #+1
        imginfo[img].append((shiftx,shifty))
                   
    
    return imginfo

##
# @brief This function groups POR vectors of different tests but same test image intoa single vector 
# @detailes Each time a test image is exhibited, one different data file is generated. This function analyses data read, identifies corresponding test images, and groups POR data of same test image.
# 
# @param file_names list of all data file names (*list of string*)
# @param lpx left eye POR x values (*list of list of floats*). The elements of this list are all the left eye POR x vectors (*list of floats*) of all tests.
# @param lpy left eye POR y values (*list of list of floats*). The elements of this list are all the left eye POR y vectors (*list of floats*) of all tests.
# @param rpx left eye POR x values (*list of list of floats*). The elements of this list are all the right eye POR x vectors (*list of floats*) of all tests.
# @param rpy left eye POR y values (*list of list of floats*). The elements of this list are all the right eye POR y vectors (*list of floats*) of all tests.
# @retval grouped_por dictionary of *list of numpy.array*, whose keys are test image names. The elements of each list are *numpy.arrays* that contain grouped POR data in the same order of the input: lpx,lpy, rpx, rpy.
#
# @remark The parameter *file_names* are used as an index to identify the POR data of each test image
def grouping_por_by_image( file_names, lpx,lpy,rpx,rpy):
    grouped_por = {}

    for file_cont in range(len(file_names)):
        each_file_name = file_names[file_cont]
        img_name = (((each_file_name.split(sep='_'))[1]).split(sep='.'))[0]
        if img_name in grouped_por:
            #print('sim')
            grouped_por[img_name][0] = grouped_por[img_name][0] + lpx[file_cont]
            grouped_por[img_name][1] = grouped_por[img_name][1] + lpy[file_cont]
            grouped_por[img_name][2] = grouped_por[img_name][2] + rpx[file_cont]
            grouped_por[img_name][3] = grouped_por[img_name][3] + rpy[file_cont]
        else:
            #print('nao')
            grouped_por[img_name]= [lpx[file_cont], lpy[file_cont], rpx[file_cont], rpy[file_cont]]
            #print(len(grouped_por),img_name,len(grouped_por[img_name][0]),type(grouped_por[img_name][0]))

    for each_one in grouped_por:
        grouped_por[each_one][0] = np.array(grouped_por[each_one][0])
        grouped_por[each_one][1] = np.array(grouped_por[each_one][1])
        grouped_por[each_one][2] = np.array(grouped_por[each_one][2])
        grouped_por[each_one][3] = np.array(grouped_por[each_one][3])
        
    return grouped_por

##
# @brief Determine POR values in image coordinates
# @details This function calculates the corresponding POR values in image coordidates from original PORs, previous calculated offsets, and image dimensions. It is necessary to plot POR data on test images. Auxiliary to *mapping_por_into_image* function.
# 
# @param x vector of POR x values
# @param y vector of POR y values
# @param shft coordinate shift (offset) with respect screen coordinates
# @param shp image size
#
# @retval ix, iy corrected POR values (with respect to image coordinates)
#
def det_indexes( x, y, shft,shp ):
    
    ix = (x-shft[0]).astype(int)
    iy = (y-shft[1]).astype(int)

    maska = ix > 0
    maskb = ix < shp[1]
    mask = maska & maskb

    ix = ix * mask

    maska = iy > 0
    maskb = iy < shp[0]
    mask = maska & maskb

    iy = iy * mask

    return ix, iy

##
# @brief Create image and matrix of mapped POR
# @detailes Create a new image from test image and original POR data. Also create a binary matrix with POR mapped into image coordinates (necessary to calculate the saliency map)
# @param image *numpy.array* containing a test image
# @param image_por list *array* containing POR data (x and y arrays, respectively)
# @shift calculated offset
# @param colour RGB vector to plot POR on image (*list of integers*)
#
# @retval por_mat POR matrix for further assessment of saliency map (*numpy.array*)
# @retval por_img test image with POR (*numpy.array*)
def mapping_por_into_image( image, image_por, shift, colour= [0,0,255]):

    por_mat = np.zeros((image.shape[0],image.shape[1]))
    por_img = np.copy(image)
    
    # Deterlinening matrix positions
    indx,indy = det_indexes( image_por[0],image_por[1], shift, image.shape )
    por_mat[indy,indx] = 1
    por_img[indy,indx] = colour
    
    return por_mat,por_img

## 
# @brief Detect fixations from POR data.
# @details 
#
# @param porx, pory POR x and y vectors (in screen coordinates)
# @param screenx, screeny x and y screen dimensions
# @param tmp time vector for each POR sample
# @param velThreshold angular velocity threshold
# @param viewingDistance viewing distance in terms of screen height (screeny)
#
# @retval fix_groups groups of fixation of one single test. It is list of dictionaries. Each element of the list corresponds to the data of one single fixation. Such data is organised into a dictionary with the following keys:
# @li @c 'x': x POR vector of the fixation
# @li @c 'y': y POR vector of the fixation
# @li @c 'timeBegin': recorded time corresponding to the beging of the fixation
# @li @c 'timeFinal': recorded time corresponding to the end of the fixation
# @li @c 'xAvg': mean of the x POR vector
# @li @c 'yAvg': mean of the y POR vector
#
def fixation_detection(porx,pory, screenx, screeny, tmp, velThreshold=50, viewingDistance = 1.5):

    # centre of the screen
    shiftx = screenx/2
    shifty = screeny/2

    # correcting PORs reference from the top-left to centre of the screen
    wPORx1 = porx - shiftx
    wPORy1 = pory - shifty

    # Calculating angles
    wPORx2 = np.roll(wPORx1,-1) # shifted versions of
    wPORx2[-1] = 0              #
    wPORy2 = np.roll(wPORy1,-1) # POR coordinates
    wPORy2[-1] = 0              #
    
    tmp2 = np.roll(tmp,-1) # shifted time vector
    tmp2[-1] = 0

    dt = tmp2-tmp
    dt = dt[:-1] * 1e-6 # convent to seconds

    
    ## method 01 - IVC reference code
    alpha1 = 2*np.arctan(np.sqrt(np.square(wPORx2-wPORx1)+np.square(wPORy2-wPORy1))/( 2 * viewingDistance * screeny ))*180/np.pi
    alpha1 = alpha1[:-1] # alpha1 is given in degrees, the last element is not considered
    vlct1 = alpha1/dt # velocity 

    ## method 02 - Author's
    dd = (viewingDistance * screeny)**2
    xx1 = wPORx1 * wPORx1 
    xx2 = wPORx2 * wPORx2
    yy1 = wPORy1 * wPORy1 
    yy2 = wPORy2 * wPORy2 

    alpha2 = np.arccos((wPORx1*wPORx2 + wPORy1*wPORy2 + dd)/(np.sqrt((xx1+yy1+dd)*(xx2+yy2+dd))))*180/np.pi
    alpha2 = alpha2[:-1]# alpha2 is given in degrees, the last element is not considered
    vlct2 = alpha2/dt

    ## Both methods seem to be equivalent for this operation conditions. Further investigations might demonstra the limits of the first approach

    
    # Detecting fixations and grouping
    counter = 0
    fix_groups = []
    Flag = False
    counter_groups = 0

    
    for counter in range(len(vlct2)):
        if vlct2[counter] < velThreshold:
            if Flag == False:
                fix_groups.append({'x': [ porx[counter],porx[counter+1] ],
                                   'y': [ pory[counter],pory[counter+1]],
                                   'timeBegin': tmp[counter],
                                   'timeFinal': tmp[counter+1],
                                   'xAvg': np.average([ porx[counter],porx[counter+1]]),
                                   'yAvg': np.average([ pory[counter],pory[counter+1]])})
                Flag = True
                
            else:
                fix_groups[counter_groups]['x'].append(porx[counter+1])
                fix_groups[counter_groups]['y'].append(pory[counter+1])
                fix_groups[counter_groups]['timeFinal']= tmp[counter+1]
                fix_groups[counter_groups]['xAvg'] = np.average(fix_groups[counter_groups]['x'])
                fix_groups[counter_groups]['yAvg'] = np.average(fix_groups[counter_groups]['y'])
        else:
            if Flag == True:
                Flag = False
                
                counter_groups += 1

    
    return fix_groups

## 
# @brief Colapse fixations, whose combined duration and visual angle distance are too short.
# @details 
#
# @param fix_groupsInp groups of fixation of one single test. It is list of dictionaries. Each element of the list corresponds to the data of one single fixation. Such data is organised into a dictionary with the following keys:
# @li @c 'x': x POR vector of the fixation
# @li @c 'y': y POR vector of the fixation
# @li @c 'timeBegin': recorded time corresponding to the beging of the fixation
# @li @c 'timeFinal': recorded time corresponding to the end of the fixation
# @li @c 'xAvg': mean of the x POR vector
# @li @c 'yAvg': mean of the y POR vector
# @param maxAngle maximum visual angle of POR that belongs to the same fixation (in degrees)
# @param maxTime  maximum time of one fixation (in ms)
# @param viewingDistance user viewing distance (in terms of screen height)
# @param screeny screen height (in pixels)
#
# @retval fix_groups groups of colapsed fixation of one single test. It is list of dictionaries. Each element of the list corresponds to the data of one single fixation. Such data is organised into a dictionary with the following keys:
# @li @c 'x': x POR vector of the fixation
# @li @c 'y': y POR vector of the fixation
# @li @c 'timeBegin': recorded time corresponding to the beging of the fixation
# @li @c 'timeFinal': recorded time corresponding to the end of the fixation
# @li @c 'xAvg': mean of the x POR vector
# @li @c 'yAvg': mean of the y POR vector
# 
def fixation_filtering1(fix_groupsInp, maxAngle = 0.5, maxTime = 75, viewingDistance = 1.5, screeny = 1024 ):
    counter_groups = 0

    fix_groups = fix_groupsInp.copy()
    
    while counter_groups < (len(fix_groups)-1):

        #print('Comprimento: ',len(fix_groups))
        #print('Contadores: ',counter_groups,counter_groups+1)
        #print('Chave1: ',fix_groups[counter_groups].keys() )
        #print('Chave2: ',fix_groups[counter_groups+1].keys() )
        #print('---')

        x2 = fix_groups[counter_groups+1]['xAvg']
        x1 = fix_groups[counter_groups]['xAvg']
        
        y2 = fix_groups[counter_groups+1]['yAvg']
        y1 = fix_groups[counter_groups]['yAvg']
        
        dd = (viewingDistance * screeny)**2

        # Elapsed time between two fixation groups
        dt = fix_groups[counter_groups+1]['timeBegin']-fix_groups[counter_groups]['timeFinal']
        dt = dt * 1e-3 # convertion from usec to msec

        # Visual angle between neighbouring fixation groups
        alpha = np.arccos((x1*x2 + y1*y2 + dd)/(np.sqrt((x1*x1+y1*y1+dd)*(x2*x2+y2*y2+dd))))*180/np.pi

        if ((dt < maxTime) & (alpha < maxAngle)):

            # Colapse x and y points
            fix_groups[counter_groups]['x'] = fix_groups[counter_groups]['x'] + fix_groups[counter_groups+1]['x']
            fix_groups[counter_groups]['y'] = fix_groups[counter_groups]['y'] + fix_groups[counter_groups+1]['y']

            # Update averages
            fix_groups[counter_groups]['xAvg'] = np.average(fix_groups[counter_groups]['x'])
            fix_groups[counter_groups]['yAvg'] = np.average(fix_groups[counter_groups]['y'])

            # Update timeFinal
            fix_groups[counter_groups]['timeFinal'] = fix_groups[counter_groups+1]['timeFinal']

            fix_groups.pop(counter_groups+1)

        else:
            counter_groups += 1

    return fix_groups

## 
# @brief Eliminate fixations, whose  duration is under a certain threshold.
# @details 
#
# @param fix_groupsInp groups of fixation of one single test. It is list of dictionaries. Each element of the list corresponds to the data of one single fixation. Such data is organised into a dictionary with the following keys:
# @li @c 'x': x POR vector of the fixation
# @li @c 'y': y POR vector of the fixation
# @li @c 'timeBegin': recorded time corresponding to the beging of the fixation
# @li @c 'timeFinal': recorded time corresponding to the end of the fixation
# @li @c 'xAvg': mean of the x POR vector
# @li @c 'yAvg': mean of the y POR vector
# 
# @param minTime minimum time to be considered a fixation (in ms)
#
# @retval fix_groups groups corrected fixation of one single test. It is list of dictionaries. Each element of the list corresponds to the data of one single fixation. Such data is organised into a dictionary with the following keys:
# @li @c 'x': x POR vector of the fixation
# @li @c 'y': y POR vector of the fixation
# @li @c 'timeBegin': recorded time corresponding to the beging of the fixation
# @li @c 'timeFinal': recorded time corresponding to the end of the fixation
# @li @c 'xAvg': mean of the x POR vector
# @li @c 'yAvg': mean of the y POR vector
# 
def fixation_filtering2(fix_groupsInp, minTime = 100 ):
    counter_groups = 0

    fix_groups = fix_groupsInp.copy()
    
    while counter_groups < len(fix_groups):

        # Fixation time
        dt = fix_groups[counter_groups]['timeFinal']-fix_groups[counter_groups]['timeBegin']
        dt = dt * 1e-3 # convertion from usec to msec

        if ( dt < minTime ):
            fix_groups.pop(counter_groups)
        else:
            counter_groups += 1

    return fix_groups

##
# @brief Calculate the filtered fixation map for every test and then group according to test image
# @detailes
#
# @param file_names vector with data file names of all tests performed
# @param lpx,lpy,rpx,rpy POR vectors (binocular data, with respect to screen coordinates)
# @param time time vector (in micro secords)
# @param screenResX, screenResY screen resolution (in pixels)
# @param viewDistance viewing distance in terms of screen height
#
# @retval grouped_fix fixation data for each image. It is a dictionary of lists, whose keys are test image names. Each element of such a list contains fixation data of one test. In turn, every test is a list of fixations, which is a dictionary with keys 'x', 'y', 'timeBegin', 'timeFinal', 'xAvg' and 'yAvg' 
#
def fixation_detection_and_grouping( file_names, lpx,lpy,rpx,rpy, time, screenResX=1280,screenResY=1024,viewDistance = 1.5):
    grouped_fix = {}
    
    for file_count in range(len(file_names)):
        each_file_name = file_names[file_count]
        img_name = (((each_file_name.split(sep='_'))[1]).split(sep='.'))[0]

        # list to array conversion
        wlpx,wlpy,wrpx,wrpy = np.array(lpx[file_count]),np.array(lpy[file_count]),np.array(rpx[file_count]),np.array(rpy[file_count])
        timeArr = np.array(time[file_count])

        # from binocular to monocular (mean of both)
        porx = (wlpx + wrpx)/2
        pory = (wlpy + wrpy)/2

        # fixation detection
        fixation_groups = fixation_detection(porx,pory,screenResX,screenResY,timeArr,velThreshold=50,viewingDistance=viewDistance)
        fixation_groups2 = fixation_filtering1(fixation_groups )
        fixation_groups3 = fixation_filtering2(fixation_groups2 )
        
        # The next line of code should be commented if not in test\study mode
        #print('User #', file_count, len(fixation_groups), len(fixation_groups2),len(fixation_groups3))
        print('Processing file #', file_count, ' from ', len(file_names))
        
        if img_name in grouped_fix:
            #print('sim')
            grouped_fix[img_name].append(fixation_groups3)
        else:
            #print('nao')
            grouped_fix[img_name]= [fixation_groups3] # all fixation data corresponding one subject for img_name
            

    
    return grouped_fix

##
# @brief Determine the saliency map from a fixation matrix.
# @detailes The saliency map is calculated by a 2D convolution between the fixation matrix and a Gaussian kernel. The standard deviation of the Gaussian function corresponds to one degree of visual angle (the viewing distance must be considered)
#
# @param fixationMat fixation matrix
# @param screen_height height of the screen in pixels
# @param viewingDistance viewing distance in terms of screen height
#
# @retval saliencyMatrix saliency map
#
def saliency_map_from_fixations( fixationMat,screen_height = 1024, viewingDistance=1.5 ):

    sigma = 2 * viewingDistance * screen_height * np.tan( 0.5 * np.pi / 180 ) # standard deviation calculation (pixels)
    saliencyMatrix = filt2D.gaussian_filter(fixationMat,sigma) # gaussian filtering
    saliencyMatrix = saliencyMatrix / saliencyMatrix.max() # normalisation
    
    return saliencyMatrix

def hotmap(image, fixmap, alpha = 0.5, gamma = 0):
    '''
    Add a given fixation map to an image.
    
    :param image: input image
    :type image: array

    :param fixmap: fixation map
    :type fixmap: array

    :param alpha: weighting parameter
    :type alpha: float

    :param gamma: dc level
    :type gamma: float

    :returns: fixation map superinposed to image (hotmap)
    :rtype: array
    '''
    return alpha*image + (1-alpha) * fixmap + gamma
# ====================================================================================   

def plot_scanpath( image, image_fix, shift, colour = [0,0,255], r = 5, line = True):
    '''
    For a given image, this function plot a scanpath based on fixations (fx,fy).

    :param image: input image
    :type image: array
  
    :param image_fix: image fixations
    :type image_fix: list of array

    :param shift: information to translate reference (from screen to image)
    :type shift: tuple
    
    :param colour: colour of fixation circles
    :type colour: list of three elements

    :param r: radius
    :type r: float

    :param line: line between fixations
    :type line: boolean

    :returns: image with scanpath
    :rtype: array
    '''
    imout = np.copy(image)
    # Deterlinening matrix positions        
    indx,indy = det_indexes( image_fix[0],image_fix[1], shift, image.shape )
    for i in range(len(indx)):
        for x in range(-r,r+1):
            for y in range(-r,r+1):
                if ( x **2 + y **2 ) <= r**2:
                    if ( (0 <= (indx[i]+x) < image.shape[0]) and (0 <= (indy[i]+y) < image.shape[1]) ):
                        imout[indx[i]+x, indy[i]+y ] = colour
                        
    if line == True:  # plot a line between fixation points
        p1 = np.empty(2) # point 1 initialisation
        p2 = np.empty(2) # point 2 initialisation
        
        k  = np.linspace(0,1,num=1000) # segment = p1 + k * (p2-p1)

        for i in range(len(indx)-1):
            # point 1 (current)
            p1[0] = indx[i] # x
            p1[1] = indy[i] # y

            if ( p1[0] in range(image.shape[0]) ) and ( p1[1] in range(image.shape[1]) ): #in image limits ?
                       
                # point 2 (next)
                p2[0] = indx[i+1] # x
                p2[1] = indy[i+1] # y

                difVector = p2 - p1

                for j in range(len(k)):
                    segment = (p1 +  k[j] * difVector).astype( int ) # conversion to integer
                    #print(segment)
                    for x in range(-1,2):
                        for y in range(-1,2):
                            if ( (segment[0]+x) in range(image.shape[0]) ) and (( segment[1]+y) in range(image.shape[1]) ): #in image limits ?
                                imout[segment[0]+x, segment[1]+y] = colour
                
    return imout
# ====================================================================================
