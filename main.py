## @package main
#
# Author: Ronaldo de Freitas Zampolo
# Begining: 07.jan.2016
#
# Read eye tracking data files in IVC format

import os

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

def extract_information(list_of_file_names, path = '/home/zampolo/engenharia/projetos/posdoc/data/ivc/eyetrack_live/etdata/'):
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
    
    for each_file in list_of_file_names:
        data_file = open(path+each_file,'r')
        time_vector = []
        rporx_vector = []
        rpory_vector = []
        lporx_vector = []
        lpory_vector = []
        type_vector = []
        
        #tv_vector = []
        for each_line in data_file:
            splited_line = each_line.split(sep = None)

            if '##' in splited_line: 
                if 'Number' in splited_line:
                    number_of_samples.append(splited_line[4])
                elif 'Area:' in splited_line:
                    calibration_area.append( (splited_line[3],splited_line[4]))
                elif 'Head' in splited_line:
                    head_distance.append(splited_line[4])

            elif ( not('Time' in splited_line) &  len(splited_line) > 0 ):
                time_vector.append(splited_line[0])
                type_vector.append(splited_line[1])
                lporx_vector.append(splited_line[17])
                lpory_vector.append(splited_line[18])
                rporx_vector.append(splited_line[19])
                rpory_vector.append(splited_line[20])
                
                #tv_vector.append(splited_line[21])

        time_matrix.append(time_vector)
        type_matrix.append(type_vector)
        lporx_matrix.append(lporx_vector)
        lpory_matrix.append(lpory_vector)
        rporx_matrix.append(rporx_vector)
        rpory_matrix.append(rpory_vector)
        #tv_matrix.append(tv_vector)
        data_file.close()
    return number_of_samples, calibration_area, head_distance, time_matrix, type_matrix, lporx_matrix, lpory_matrix, rporx_matrix, rpory_matrix #, tv_matrix

# test
files = find_files()
ns, ca, hd, ti, tp, lporx,lpory,rporx,rpory = extract_information(files)
print('Number of samples: ',ns[0])
print('Calibration area: ',ca[0])
print('Head distance: ',hd[0])
print('Number of time points: ',len(ti[0]))
#print('Size: ',s[0])
#print('TV: ',tv[0])
