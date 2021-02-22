from PIL import Image
import glob
import os
for f in glob.glob("*/*.png"):
    im = Image.open(f)
    rgb_im = im.convert("RGB")
    rgb_im.save(f.replace("png","jpeg"),quality=100,subsampling=0)
    im.close()
    os.remove(f)