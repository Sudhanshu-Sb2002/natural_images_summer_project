import numpy as np
import mat73
from PIL import Image
path="D:\\OneDrive - Indian Institute of Science\\5th Sem\\Summer project\\Data\\simple_data_v1\\RGB_texture_images.mat"
rawdata = mat73.loadmat(path)
data =(rawdata["images"]*255).astype(np.uint8)

for i in range (2):
    img = Image.fromarray(data[i],'RGB')
    img.show()

