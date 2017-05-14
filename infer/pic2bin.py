from PIL import Image
import numpy as np

'''
width= 24
height=24
im = Image.open("c919.jpg")
out = im.resize((width, height),Image.ANTIALIAS)

out.save("alexout.jpg")
'''
width= 32
height= 32
filename='airplane2.png'
print("-->>转换 %s" % filename)
im = Image.open(filename) #out.jpg
im = (np.array(im))

r = im[:,:,0].flatten()
g = im[:,:,1].flatten()
b = im[:,:,2].flatten()
label = [1]

out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
out.tofile("alexout.bin")