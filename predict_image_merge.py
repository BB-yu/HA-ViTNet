import os
import sys

from osgeo import gdal
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from model import *
from loss import *
from PIL import Image
import torch
import os
import sys

import numpy as np

from dataload import LoadData,datapre,datapre_array

import skimage

from skimage import morphology
import os
from PIL import Image
import matplotlib.pyplot as plt


def writeTiff(im_data, im_geotrans, im_proj, path):

    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_Int16
    elif 'int32' in im_data.dtype.name:
        datatype = gdal.GDT_Int32
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)

    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)

        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del dataset


def CoordTransf(Xpixel, Ypixel, GeoTransform):
    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    return XGeo, YGeo



def TifCrop(TifPath, SavePath, CropSize, RepetitionRate,label=None,validdataremove=False,predict=True):
    
    # TifPath=infiletif
    print("------------------------------------------")
    CropSize = int(CropSize)
    RepetitionRate = float(RepetitionRate)
    dataset_img = gdal.Open(TifPath)
    if dataset_img == None:
        print(TifPath + "")


    # if not os.path.exists(SavePath):
    #     os.makedirs(SavePath)

    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    bands = dataset_img.RasterCount
    print("", height)
    print("", width)
    print("", bands)

    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    img = dataset_img.ReadAsArray(0, 0, width, height)

    data_bands = img
    if label is not None:
        label_band = label
        print(f'label：{label_band.shape}')
    # import matplotlib.pyplot as plt
    # from rasterio.plot import  reshape_as_image
    # plt.imshow(reshape_as_image(data_bands)[:,:,[2,1,0]])
    


    RowNum = int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))

    ColumnNum = int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
    print("", RowNum)
    print("", ColumnNum)


    if predict:
        new_name=1
    else:
        new_name = len(os.listdir(SavePath)) + 1
    
    print('new_name',new_name)
    predictimages={}
    

    for i in range(RowNum):
        for j in range(ColumnNum):
            cropped_data = data_bands[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            if label is not None:
                cropped_label = label_band[
                            int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                            int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
                if validdataremove:
                    if np.sum(cropped_label)==0:continue

            

            XGeo, YGeo = CoordTransf(int(j * CropSize * (1 - RepetitionRate)),
                                     int(i * CropSize * (1 - RepetitionRate)),
                                     geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
            if predict:
                predictimages[new_name]=cropped_data
            else:

                writeTiff(cropped_data, crop_geotrans, proj, os.path.join(SavePath, "data_%d.tif" % new_name))
            if label is not None:

                writeTiff(cropped_label, crop_geotrans, proj, os.path.join(SavePath, "label_%d.tif" % new_name))


            new_name = new_name + 1


    for i in range(RowNum):
        cropped_data = data_bands[:,
                      int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize): width]
        if label is not None:
            cropped_label = label_band[
                        int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                        (width - CropSize): width]
            if validdataremove:
                if np.sum(cropped_label)==0:continue

        XGeo, YGeo = CoordTransf(width - CropSize,
                                 int(i * CropSize * (1 - RepetitionRate)),
                                 geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
        if predict:
            predictimages[new_name]=cropped_data
        else:

            writeTiff(cropped_data, crop_geotrans, proj, os.path.join(SavePath, "data_%d.tif" % new_name))

        if label is not None:
            writeTiff(cropped_label, crop_geotrans, proj, os.path.join(SavePath, "label_%d.tif" % new_name))

        new_name = new_name + 1


    for j in range(ColumnNum):
        cropped_data = data_bands[:,
                      (height - CropSize): height,
                      int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        if label is not None:
            cropped_label = label_band[
                        (height - CropSize): height,
                        int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            if validdataremove:
                if np.sum(cropped_label)==0:continue

        XGeo, YGeo = CoordTransf(int(j * CropSize * (1 - RepetitionRate)),
                                 height - CropSize,
                                 geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
        if predict:
            predictimages[new_name]=cropped_data
        else:

            writeTiff(cropped_data, crop_geotrans, proj, os.path.join(SavePath, "data_%d.tif" % new_name))

        if label is not None:
            writeTiff(cropped_label, crop_geotrans, proj, os.path.join(SavePath, "label_%d.tif" % new_name))


        new_name = new_name + 1

    cropped_data = data_bands[:,
                  (height - CropSize): height,
                  (width - CropSize): width]
    if label is not None:
        cropped_label = label_band[
                    (height - CropSize): height,
                    (width - CropSize): width]

    XGeo, YGeo = CoordTransf(width - CropSize,
                             height - CropSize,
                             geotrans)
    crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])
    if predict:
        predictimages[new_name]=cropped_data
    else:

        writeTiff(cropped_data, crop_geotrans, proj, os.path.join(SavePath, "data_%d.tif" % new_name))

    if label is not None:
        writeTiff(cropped_label, crop_geotrans, proj, os.path.join(SavePath, "label_%d.tif" % new_name))

    new_name = new_name + 1

    if predict:
        return predictimages




def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_Uint16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def CoordTransf(Xpixel, Ypixel, GeoTransform):
    XGeo = GeoTransform[0] + GeoTransform[1] * Xpixel + Ypixel * GeoTransform[2]
    YGeo = GeoTransform[3] + GeoTransform[4] * Xpixel + Ypixel * GeoTransform[5]
    return XGeo, YGeo
    

def TifStitch(OriTif, predic_res, ResultPath, RepetitionRate):
    RepetitionRate = float(RepetitionRate)
    print("-------------------------------------------")
    dataset_img = gdal.Open(OriTif)
    
    # import rasterio 
    
    # src=rasterio.open(OriTif)
    
    width = dataset_img.RasterXSize
    height = dataset_img.RasterYSize
    bands = dataset_img.RasterCount
    proj = dataset_img.GetProjection()
    geotrans = dataset_img.GetGeoTransform()
    # ori_img = dataset_img.ReadAsArray(0, 0, width, height)
    bands=1
    print("波段数为：", bands)
    
    

    if bands == 1:
        shape = [height, width]
    else:
        shape = [bands, height, width]
    result = np.zeros(shape, dtype='uint8')


    NameArray=list(predic_res.keys())

    OriImgArray=list(predic_res.values())
    
    height_crop,width_crop=OriImgArray[0].shape
    bands_crop=1

    
    
    print("：", len(OriImgArray))

    
    

    RowNum = int((height - height_crop * RepetitionRate) / (height_crop * (1 - RepetitionRate)))

    ColumnNum = int((width - width_crop * RepetitionRate) / (width_crop * (1 - RepetitionRate)))

    sum_img = RowNum * ColumnNum + RowNum + ColumnNum + 1
    print("", RowNum)
    print("", ColumnNum)
    print("", sum_img)


    if bands_crop == 1:
        shape_crop = [height_crop, width_crop]
    else:
        shape_crop = [bands_crop, height_crop, width_crop]
    img_crop = np.zeros(shape_crop)

    ImgArray = []
    count = 0
    for i in range(sum_img):
        img_name = i + 1
        for j in range(len(OriImgArray)):
            if img_name == int(NameArray[j]):
                image = OriImgArray[j]
                count = count + 1
                break
            else:
                image = img_crop
        if len(image.shape)==3:
            ImgArray.append(np.transpose(image,(2,0,1)))
        elif len(image.shape)==2:
            ImgArray.append(image)

    print("", count)
    print("", len(ImgArray))



    num = 0
    for i in range(RowNum):
        for j in range(ColumnNum):

            if (bands == 1):
                result[int(i * height_crop * (1 - RepetitionRate)): int(i * height_crop * (1 - RepetitionRate)) + height_crop,
                            int(j * width_crop * (1 - RepetitionRate)): int(j * width_crop * (1 - RepetitionRate)) + width_crop] = ImgArray[num]

            else:
                result[:,
                            int(i * height_crop * (1 - RepetitionRate)): int(i * height_crop * (1 - RepetitionRate)) + height_crop,
                            int(j * width_crop * (1 - RepetitionRate)): int(j * width_crop * (1 - RepetitionRate)) + width_crop] = ImgArray[num]
            num = num + 1

    for i in range(RowNum):
        if (bands == 1):
            result[int(i * height_crop * (1 - RepetitionRate)): int(i * height_crop * (1 - RepetitionRate)) + height_crop,
                      (width - width_crop): width] = ImgArray[num]
        else:
            result[:,
                      int(i * height_crop * (1 - RepetitionRate)): int(i * height_crop * (1 - RepetitionRate)) + height_crop,
                      (width - width_crop): width] = ImgArray[num]
        num = num + 1

    for j in range(ColumnNum):
        if (bands == 1):
            result[(height - height_crop): height,
                      int(j * width_crop * (1 - RepetitionRate)): int(j * width_crop * (1 - RepetitionRate)) + width_crop] = ImgArray[num]
        else:
            result[:,
                      (height - height_crop): height,
                      int(j * width_crop * (1 - RepetitionRate)): int(j * width_crop * (1 - RepetitionRate)) + width_crop] = ImgArray[num]
        num = num + 1

    if (bands == 1):
        result[(height - height_crop): height,
                        (width - width_crop): width] = ImgArray[num]
    else:
        result[:,
                    (height - height_crop): height,
                    (width - width_crop): width] = ImgArray[num]
    num = num + 1

    writeTiff(result, geotrans, proj, ResultPath)
def predictimage(img):
    logits_mask=model(img.to('cuda', dtype=torch.float32).unsqueeze(0))
    
    # pred_mask=torch.sigmoid(logits)
    # predict = (pred_mask>0.5)*1.0
    pred_mask=torch.sigmoid(logits_mask)
    pred_mask=(pred_mask >ratio)*1.0
    pre=pred_mask.detach().cpu().numpy().squeeze(0).squeeze(0).astype(np.uint8)
    # plt.imshow(pre,'gray')
    # plt.show()
    selem=skimage.morphology.disk(3) 
    pre[pre==1]=255
    return pre

if __name__ == '__main__':
    # infiletif=r'D:\lab\lab7\JL1KF01A_PMS06_20211214112242_200069276_101_0007_GS.sharpening'

    infiletif=r''
    
    files=glob.glob(r"D*/label*.hdr")[6:]

    for file in files:
        
        fi=[file.replace(".hdr",""),file.replace(".hdr",".tif")]
        
        needfile=[s for s in fi if os.path.exists(s)][0]

        
        infiletif=needfile
        
        da=needfile.split('\\')[2]
        outptfile=rf'efficientnet-b4_imagenet_{da}.pt'
        name=outptfile.replace('.pt','.tif')
        
        
        ResultPath=rf'D:\lab\{da}\{name}'
        outdirtif=None
        # infilemasktif=r'D:\lab\lab7/label'
        
        RepetitionRate=0.2
        ratio=0.5
        # os.makedirs(outdirtif,exist_ok=True)
        # dataset_img = gdal.Open(infilemasktif)
        # label = dataset_img.ReadAsArray()[-1]
        checkpoint_path=outptfile
        
        device = 'cuda'
        model = SegmentationModel()
        model = model.to(device)
        model.load_state_dict(torch.load(checkpoint_path))
        
        
        
        predictimages=TifCrop(infiletif, outdirtif, 256, RepetitionRate,label=None,validdataremove=False,predict=True)
        torch.cuda.is_available()

        # checkpoint_path = r"best_model_deeplab++1_709.pt"
        

        model.to(device)
        model.eval()
        predic_res={}
        kys=list(predictimages.keys())
        for ky in tqdm(kys[:]):

                img=np.transpose(predictimages[ky],(1,2,0))

                img=datapre_array(img).getite()

                predic_im=predictimage(img) 

                predic_res[ky]=predic_im
        
        TifStitch(infiletif, predic_res, ResultPath,RepetitionRate)

