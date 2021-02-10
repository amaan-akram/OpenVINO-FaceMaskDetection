import os
import re

mask_path = 'mask/'
no_mask_path = 'no_mask/'

for image in os.listdir('lfw_masked'):
    checkImage = (re.findall('\d+', image))[0]
    if (int(checkImage)%2 == 0):
        os.rename('lfw_masked/' + image, mask_path + image)
    else:
        os.rename('lfw_masked/' + image, no_mask_path + image)


    print(checkImage)
