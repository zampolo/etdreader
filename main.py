## @package main
#
# Author: Ronaldo de Freitas Zampolo
# Begining: 07.jan.2016
#
# Read eye tracking data files in IVC format

import os
import numpy as np
import matplotlib.pyplot as plt

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

            # data portion of the file
            elif ( not('Time' in splited_line) &  len(splited_line) > 0 ):
                # time information
                time_vector.append(splited_line[0])
                # type of data (sample or message)
                type_vector.append(splited_line[1])
                # POR
                lporx_vector.append(float(splited_line[17]))
                lpory_vector.append(float(splited_line[18]))
                rporx_vector.append(float(splited_line[19]))
                rpory_vector.append(float(splited_line[20]))
                
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
    
    return number_of_samples, calibration_area, head_distance, time_matrix, type_matrix, lporx_matrix, lpory_matrix, rporx_matrix, rpory_matrix #, tv_matrix

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
        shiftx = (screen_size[0] - imginfo[img][0][0])//2 +1
        shifty = (screen_size[1] - imginfo[img][0][1])//2 +1
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

    for name_cont in range(len(file_names)):
        each_file_name = file_names[name_cont]
        img_name = (((each_file_name.split(sep='_'))[1]).split(sep='.'))[0]
        if img_name in grouped_por:
            #print('sim')
            grouped_por[img_name][0] = grouped_por[img_name][0] + lpx[name_cont]
            grouped_por[img_name][1] = grouped_por[img_name][1] + lpy[name_cont]
            grouped_por[img_name][2] = grouped_por[img_name][2] + rpx[name_cont]
            grouped_por[img_name][3] = grouped_por[img_name][3] + rpy[name_cont]
        else:
            #print('nao')
            grouped_por[img_name]= [lpx[name_cont], lpy[name_cont], rpx[name_cont], rpy[name_cont]]
            #print(len(grouped_por),img_name,len(grouped_por[img_name][0]),type(grouped_por[img_name][0]))

    for each_one in grouped_por:
        grouped_por[each_one][0] = np.array(grouped_por[each_one][0])
        grouped_por[each_one][1] = np.array(grouped_por[each_one][1])
        grouped_por[each_one][2] = np.array(grouped_por[each_one][2])
        grouped_por[each_one][3] = np.array(grouped_por[each_one][3])
        
    return grouped_por

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
    


def mapping_por_into_image( image, image_por, shift, lcolour='blue', rcolour='red' , mcolour= 'green'):

    lpor_mat = np.zeros(image.shape)
    rpor_mat = np.zeros(image.shape)
    mpor_mat = np.zeros(image.shape)

    lpor_img = np.copy(image)
    rpor_img = np.copy(image)
    mpor_img = np.copy(image)

    # Left POR
    indx,indy = det_indexes( image_por[0],image_por[1], shift, image.shape )
    lpor_mat[indy,indx] = 1
    lpor_img[indy,indx] = [0,0,255]
    # Right POR
    indx,indy = det_indexes( image_por[2],image_por[3], shift, image.shape )
    rpor_mat[indy,indx] = 1
    rpor_img[indy,indx] = [255,0,0]
    # Mean POR
    indx,indy = det_indexes( 0.5*(image_por[0]+image_por[2]),0.5*(image_por[1]+image_por[3]), shift, image.shape )
    mpor_mat[indy,indx] = 1
    mpor_img[indy,indx] = [0,255,0]
    
    return lpor_mat, rpor_mat, mpor_mat, lpor_img, rpor_img, mpor_img


# ========== test ================

files = find_files()
ns, ca, hd, ti, tp, lporx,lpory,rporx,rpory = extract_information(files)
#print(' ================================== ')
#print('Number of samples: ',ns[0])
#print('Calibration area: ',ca[0],' pixels')
#print('Head distance: ',hd[0], ' mm')
#print('Number of time points: ',len(ti[0]))


imageinfo = load_image_information(screen_size=ca[0])
#print('Image information size: ', len(imageinfo))
#print(' ================================== ')

#for item in imageinfo:
#    print(item,'- resolution: ',imageinfo[item][0],' shift: ',imageinfo[item][1])

gp = grouping_por_by_image( files, lporx, lpory, rporx, rpory)
#print(len(gp),len(gp['stream'][0]),type(gp['stream'][0]))

#print('Image information: ', imageinfo)
#print('Size: ',s[0])
#print('TV: ',tv[0])

path_images = '/home/zampolo/engenharia/projetos/posdoc/data/ivc/eyetrack_live/images/'
image_name = 'lighthouse2'
img = plt.imread(path_images+image_name+'.bmp')

lmat, rmat, mmat, limg, rimg, mimg = mapping_por_into_image( image = img, image_por=gp[image_name], shift=imageinfo[image_name][1])


plt.imshow(img)
plt.title('Original image: '+ image_name )

#plt.figure()
#plt.imshow(lmat)
#plt.title(image_name+': left eye POR')

plt.figure()
plt.imshow(limg)
plt.title(image_name+': left eye POR on image')

#plt.figure()
#plt.imshow(rmat)
#plt.title(image_name+': right eye POR')

plt.figure()
plt.imshow(rimg)
plt.title(image_name+': right eye POR on image')

#plt.figure()
#plt.imshow(mmat)
#plt.title(image_name+': average POR')

plt.figure()
plt.imshow(mimg)
plt.title(image_name+': average POR on image')

plt.show()

