from . import detect
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def add_box(path, username):
    os.chdir('processing/Image_Detection')
    new_path = detect.run(weights='Mymodel/best.pt', source=path, project='../../media/images/detect_img')
    os.chdir('../..')
    new_path[0] = new_path[0].replace("\\", '/')
    new_path[0] = new_path[0][new_path[0].index('/media'):]
    return new_path

pathlib.PosixPath = temp