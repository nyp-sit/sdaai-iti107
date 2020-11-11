""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-x XMLDIR] [-o OUTPUTDIR] [-r RATIO] 

Partition dataset of images into training and testing sets

"""
import os
import re
from shutil import copyfile
import argparse
import math
import random
import numpy as np

def createOutputDir(outputDir): 
    # the following subfolders will be created in the target outputDIr
    # test/images, test/annotations, train/images, train/annotations
    test_dir = os.path.join(outputDir, 'test')
    train_dir = os.path.join(outputDir, 'train')
    test_image_dir = os.path.join(outputDir, 'test', 'images')
    test_annot_dir = os.path.join(outputDir, 'test', 'annotations')
    train_image_dir = os.path.join(outputDir, 'train', 'images')
    train_annot_dir = os.path.join(outputDir, 'train', 'annotations')
   
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_image_dir):
        os.mkdir(test_image_dir)
    if not os.path.exists(test_annot_dir):
        os.mkdir(test_annot_dir)
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_annot_dir):
        os.mkdir(train_annot_dir) 

    return test_image_dir, test_annot_dir, train_image_dir, train_annot_dir
    

def iterate_dir(imageDir, xmlDir, outputDir, ratio):
    # imageDir = imageDir.replace('\\', '/')
    # xmlDir = xmlDir.replace('\\', '/')
    # outputDir = outputDir.replace('\\','/')

    test_image_dir, test_annot_dir, train_image_dir, train_annot_dir = createOutputDir(outputDir)

    images = [f for f in os.listdir(imageDir)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)


    indices = np.random.permutation(num_images)
    test_indices = indices[:num_test_images]
    train_indices = indices[num_test_images:]

    for idx in test_indices:
        fname = images[idx]
        xml_fname = os.path.splitext(fname)[0]+'.xml'
        print('copy {} to test dir'.format(fname))
        copyfile(os.path.join(imageDir, fname),
                os.path.join(test_image_dir, fname))
        copyfile(os.path.join(xmlDir, xml_fname),
                os.path.join(test_annot_dir, xml_fname))
        
    for idx in train_indices:
        fname = images[idx]
        xml_fname = os.path.splitext(fname)[0]+'.xml'
        print('copy {} to train dir'.format(fname))
        copyfile(os.path.join(imageDir, fname),
                os.path.join(train_image_dir, fname))
        copyfile(os.path.join(xmlDir, xml_fname),
                os.path.join(train_annot_dir, xml_fname))
    

def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image files is stored.',
        type=str,
        default=None
    )

    parser.add_argument(
        '-x', '--xmlDir',
        help='Path to the folder where the XML annotation files is stored.',
        type=str,
        default=None
    )

    parser.add_argument(
        '-o', '--outputDir',
        help='Path to the output folder where the train and test dirs should be created.  ',
        type=str,
        default=None
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.1,
        type=float)
    
    args = parser.parse_args()

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.xmlDir, args.outputDir, args.ratio)


if __name__ == '__main__':
    main()