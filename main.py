# main.py
#
# Ronaldo de Freitas Zampolo
# 07.jan.2016
#
# Read eye tracking data files in IVC format

import os

def find_files( path = '/home/zampolo/engenharia/projetos/posdoc/data/ivc/eyetrack_live/etdata/', extentions = ['txt'] ):
    found_files = os.listdir(path)
    for each_file in found_files:
        file_type = os.path.splitext(each_file)
        if (not(file_type in extentions)):
            found_files.remove(each_file)
        
    return found_files
#-----------------

# test
files = find_files()
