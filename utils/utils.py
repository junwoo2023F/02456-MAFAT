# IMPORTS
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


# VARIABLES
class MAFAT_Dataset(Dataset):
    """MAFAT dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
#         self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = getFileName(name=str(self.metadata.iloc[idx, 1]),filesPath=self.root_dir)
        image = io.imread(img_name)
        xyPoints = self.metadata.iloc[idx, 2:10].values #xy-coordinates, a maxiumum of 4 objects per image
        xyPoints = xyPoints.astype('float').reshape(-1, 2)
        xyRaw = self.metadata.iloc[idx, 2:10].values
        xyRaw = xyRaw.astype('float')
        sample = {'image': image, 'xypoints': xyPoints,'xyRaw': xyRaw,'filename': str(self.metadata.iloc[idx, 1]),
                 'tagid': str(self.metadata.iloc[idx, 0]),'OOB_object': 'None'}
        
        return sample



def getFileName(filesPath = None,name = None):
    """
    Returns the full name of image given a path and unique string identifying file
    """
    for root, dirs, files in os.walk(filesPath):
        name = str(name) #ensure
        filename = "".join(s for s in files if name in s) #get filename in folder
        return os.path.join(root, filename) #get full string
    
def show_landmarks(image, xypoints):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(xypoints[:, 0], xypoints[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.figure()
    plt.show()

def cropIm(im,xvec,yvec,pad,imExtent):
    """
        Crop image
    """
    assert pad >= 0, "Padding must be positive"
    boxlowx = int(min(xvec)-pad)
    boxlowy = int(min(yvec)-pad)
    boxhix = int(max(xvec)+pad)
    boxhiy = int(max(yvec)+pad)
    im = im[boxlowy:boxhiy,boxlowx:boxhix,:] #Ensuring images are square
    return im    

def cropPolygonImage(image,xvec,yvec,pad,root,filename,pltIm=None,fixedshape=64):
    """
        Crops image based on set of coordinates
    """
    from PIL import Image,ImageDraw
    assert image.shape[2] >= 3, 'Image requires alpha layer in forth channel' 
    polygon = [(xvec[0],yvec[0]),(xvec[1],yvec[1]),(xvec[2],yvec[2]),(xvec[3],yvec[3])]
    maskIm = Image.new('L', (image.shape[1], image.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)

    # assemble new image (uint8: 0-255)
    newImArray = np.empty(image.shape,dtype='uint8')

    # colors (three first columns, RGB)
    newImArray[:,:,:3] = image[:,:,:3]

    # transparency (4th column)
    newImArray[:,:,3] = mask*255

    # back to Image from numpy
    croppedImage = cropIm(newImArray,xvec=xvec,yvec=yvec,pad=pad,imExtent=fixedshape)
    newIm = Image.fromarray(croppedImage, "RGBA")
    newIm = newIm.resize((256,256),Image.BICUBIC)
    newIm.save(osp.join(root,'cropped/',filename))
    if pltIm is not None:
        plt.imshow(croppedImage)
        plt.show()
    
dataset = MAFAT_Dataset(csv_file='./dataset_v2/train.csv',
                                    root_dir='dataset_v2/root/train/')

def copyToClassFolders(listOfClasses,rootPath=None,df=None):
    """
        Copies cropped images and assigns them to folder representative of that feature/class
    """
    import shutil
    import difflib
    list_of_cropped_images=os.listdir(rootPath+'cropped/')
    ims = 0
    errorIm=0
    classCols=df.columns[12:-1]
    for image in list_of_cropped_images: # loop over the cropped images
        s = str(image)
        s = s.replace('.png','')
        s = s.split('_') #split into image_id and tag_id
        try:
            dataRow = df[df.tag_id == int(s[1])]
        except:
            print('unexpected file type')
            continue
        if not dataRow.empty:
            feat = dataRow #get only the  class features
            for i in range(10,25):
                val = feat.iloc[0,i]
                if isinstance(val,str) is True: #type of general class
                    shutil.copy2(rootPath+'cropped/'+image, rootPath+'classes/'+val) 
                if i > 11  and i < 24:            
                    if int(val) > 0:
                        item = i-12
                        match = difflib.get_close_matches(str(classCols[item]),listOfClasses,n=1)[0]
                        shutil.copy2(rootPath+'cropped/'+image, rootPath+'classes/'+match)
        else:
            errorIm+=1
            if errorIm % 100 ==0:
                print(errorIm)
        ims+=1
        if ims % 250 == 0:
            print(f'Images copied: {ims}')                 

