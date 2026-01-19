import os
import sys
try:
    import gdal
except:
    from osgeo import gdal
import numpy as np
from PIL import Image


from osgeo import gdal
import numpy as np

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



def TifCrop(TifPath, SavePath, CropSize, RepetitionRate,label=None,validdataremove=False):
    
    # TifPath=infiletif
    print("-------------------------------------------")
    CropSize = int(CropSize)
    RepetitionRate = float(RepetitionRate)
    dataset_img = gdal.Open(TifPath)
    if dataset_img == None:
        print(TifPath + " ")

    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

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

    


    RowNum = int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))

    ColumnNum = int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))
    print("", RowNum)
    print("", ColumnNum)


    new_name = len(os.listdir(SavePath)) + 1
    print('new_name',new_name)

    

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
                    if np.sum(cropped_label)==0:
                        new_name+=1
                        continue

            

            XGeo, YGeo = CoordTransf(int(j * CropSize * (1 - RepetitionRate)),
                                     int(i * CropSize * (1 - RepetitionRate)),
                                     geotrans)
            crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])

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
                if np.sum(cropped_label)==0:
                    new_name+=1
                    continue

        XGeo, YGeo = CoordTransf(width - CropSize,
                                 int(i * CropSize * (1 - RepetitionRate)),
                                 geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])


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
                if np.sum(cropped_label)==0:
                    new_name+=1
                    continue

        XGeo, YGeo = CoordTransf(int(j * CropSize * (1 - RepetitionRate)),
                                 height - CropSize,
                                 geotrans)
        crop_geotrans = (XGeo, geotrans[1], geotrans[2], YGeo, geotrans[4], geotrans[5])


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


    writeTiff(cropped_data, crop_geotrans, proj, os.path.join(SavePath, "data_%d.tif" % new_name))

    if label is not None:
        writeTiff(cropped_label, crop_geotrans, proj, os.path.join(SavePath, "label_%d.tif" % new_name))

    new_name = new_name + 1


if __name__ == '__main__':

        
    infiletif=r' '
    outdirtif=os.path.splitext(infiletif)[0]+'image/'
    
    # outdirtif=r"D:\lab\data_label\lab7_8/image"
    os.makedirs(outdirtif,exist_ok=True)

    infilemasktif=r'D:\lab\lab3\导出/labelnew'
    dataset_img = gdal.Open(infilemasktif)
    label = dataset_img.ReadAsArray()[-1]
    
    # outdirmask=r"D:\lab\data_label\lab7_8/label"
    
    TifCrop(infiletif, outdirtif, 256, 0.2,label=label,validdataremove=False)



