import os
import shutil
import datetime
import random
maskDIR = 'no_maskFull'


now = 0



for i in range(11000):
    Allfiles = os.listdir(maskDIR)
    numberOfFiles = len(Allfiles)
    randomIndex = random.randint(0, numberOfFiles)
    print(Allfiles[randomIndex])
    shutil.move(maskDIR + "/" + Allfiles[randomIndex], "no_maskForSMFD/" + Allfiles[randomIndex])
    numberOfFiles = numberOfFiles - 1




