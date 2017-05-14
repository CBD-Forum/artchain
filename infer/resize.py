#import Image
from PIL import Image

width= 32
height=32
im = Image.open("j20.jpg")#c919.jpg cat1.png
out = im.resize((width, height),Image.ANTIALIAS)

out.save("j202.jpg")