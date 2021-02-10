import os
import shutil
import datetime

maskDIR = 'mask'


now = 0

Allfolders = os.listdir(maskDIR)
for folder_name in Allfolders:
    folder = maskDIR + "/" + folder_name
    files = os.listdir(folder)
    for file in files:
        now = now+1

        oldext = os.path.splitext(file)[1]
        os.rename(folder + "/" + file, maskDIR + "/" + str(now) + oldext)




