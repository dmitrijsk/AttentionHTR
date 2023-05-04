import fire
import os
import lmdb
import cv2
import numpy as np
import re

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(inputPath, gtFile, outputPath, checkValid: bool=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        gtFile     : list of image path and label
        outputPath : LMDB output path
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            with open(os.path.join(outputPath, 'error_image_log.txt'), 'a', encoding='utf-8') as log:
                log.write('%s does not exist\n' % imagePath)
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid and not checkImageIsValid(imageBin):
            with open(os.path.join(outputPath, 'error_image_log.txt'), 'a', encoding='utf-8') as log:
                log.write('%s is not a valid image\n' % imagePath)
            pass

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            with open(os.path.join(outputPath, 'log.txt'), 'a', encoding='utf-8') as log:
                log.write('%s-th image data occurred error\n' % str(i))
        cnt += 1
    
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
   fire.Fire(createDataset)
