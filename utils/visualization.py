import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def txt2numpy(root_dir,height = 256,width = 189,imshow = True):
    files = os.listdir(root_dir)
    for f in files:
        s = []
        test = open(os.path.join(root_dir,f)).readlines()
        for i in range(len(test)):
            line = map(float,test[i].strip().split())
            s.append(line)
        img = np.zeros((height,width))
        for i in range(len(s)):
            for j in range(len(s[i])):
                img[i,j] = s[i][j]
        plt.imshow(img)
        plt.title(f)
        plt.pause(1)
    
        