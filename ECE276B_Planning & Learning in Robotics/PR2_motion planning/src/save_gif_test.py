# -*- coding: utf-8 -*-
"""
Created on Sun May  8 22:06:48 2022

@author: Yen
"""

import os
import imageio

png_dir = 'gif'
images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('motion.gif', images, fps = 10)