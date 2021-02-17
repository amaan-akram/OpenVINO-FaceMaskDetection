import os
import shutil
import datetime
import random


AllmaskDIR = '../ALL_DATA/mask'

now = 0

AllfilesList = os.listdir('../lfw_masked/data/mask')
for i in range(11000):
    Allfiles = os.listdir(AllfilesList)
    numberOfFiles = len(Allfiles)
    randomIndex = random.randint(0, numberOfFiles)
    print(Allfiles[randomIndex])
    shutil.move(maskDIR + "/" + Allfiles[randomIndex], "no_maskForSMFD/" + Allfiles[randomIndex])
    numberOfFiles = numberOfFiles - 1
