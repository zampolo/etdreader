## @package etdreader
#
# Author: Ronaldo de Freitas Zampolo
# Begining: 07.jan.2016
# Version: 1.0 (14.jan.2016)
#
# Read eye tracking data files in IVC format

import os
import numpy as np
import scipy.signal as sig
import scipy.ndimage.filters as filt2D

## find_files
#
# found_files = find_files(path: string,types: list)
# inputs:
#    path: string
#    types: list of strings
#
# output:
#   found_files: list of strings containing all the files of a given path of type 
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

# extract_information
#
# extract_information(list_of_file_names, path)
# inputs:
#   list_of_fil_names: list of strings
#   path: string, default value '/home/zampolo/engenharia/projetos/posdoc/data/ivc/eyetrack_live/etdata/'
#
# outputs:
#   number_of_samples: string
#   calibration_area: tuple of strings
#   head_distance: string
#   time_matrix: matrix (line: user, column: sampling time)
#   type_matrix: data type (message ou sample) (line: user, column: sampling time)
#   lporx_matrix: left point od regard (por) x (line: user, column: sampling time)
#   lpory_matrix: left por y (line: user, column: sampling time)
#   rporx_matrix: right por x (line: user, column: sampling time)
#   rpory_matrix: right por y (line: user, column: sampling time)
#
#   Obs.: all the outputs are strings. Perhaps in the near future, internal convertions to real ou integer types will be implemented.
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
    
    #tv_matrix = []

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
        
        #tv_vector = []

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
        #tv_matrix.append(tv_vector)
        
        # closing data file
        data_file.close()
    
    return number_of_samples, calibration_area, head_distance, sampling_rate, time_matrix, type_matrix, lporx_matrix, lpory_matrix, rporx_matrix, rpory_matrix #, tv_matrix

# load_image_information
#
# load_image_information(screen_size,name)
# 
# input:
#   screen_size: dimensions of the screen (tuple)
#   name: name of data base of interest (string)
#
# output:
#   imginfo : dictionary {name_of_the_images: [(resolution),(shift)]}
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

# grouping_por_by_image
#
# grouping_por_by_image( image_names, file_names, por)
# 
# inputs:
#   image_names: list of string
#   file_names: list of srting
#   por: list of integer [lporx,lpory,rpox,rpory]
#
# output:
#   grouped_por: dictionary of matrix {image_names: [lporx,lpory,rporx,rpory]}
#
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

# det_indexes
#
# det_indexes( x, y, shft,shp )
# 
# inputs:
#   x,y: POR coordinate vectors
#   shft: coordinate shift with respect screen coordinates
#   shp: image size
#
# output:
#   ix, iy: corrected coordinates
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

# mapping_por_into_image
#
# mapping_por_into_image( image, image_por, shift, lcolour='blue', rcolour='red' , mcolour= 'green')
#
# Inputs:
#
#
# Outputs:
#
#
def mapping_por_into_image( image, image_por, shift, colour= [0,0,255]):

    por_mat = np.zeros(image.shape)
    por_img = np.copy(image)
    
    # Deterlinening matrix positions
    indx,indy = det_indexes( image_por[0],image_por[1], shift, image.shape )
    por_mat[indy,indx] = 1
    por_img[indy,indx] = colour
    
    return por_mat,por_img

# detect fixations
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


def saliency_map_from_fixations( fixationMat,screen_height = 1024, viewingDistance=1.5 ):

    sigma = 2 * viewingDistance * screen_height * np.tan( 0.5 * np.pi / 180 ) # standard deviation calculation (pixels)
    saliencyMatrix = filt2D.gaussian_filter(fixationMat,sigma) # gaussian filtering
    saliencyMatrix = saliencyMatrix / saliencyMatrix.max() # normalisation
    
    return saliencyMatrix

