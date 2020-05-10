# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:34:44 2020

@author: sakib

"""



import os
from zipfile import ZipFile
import urllib.request


def change_base_dir(base_dir_path):
    """ Change the working directopry of the code"""
    
    if not os.path.exists(base_dir_path):
        print ('creating directory', base_dir_path)
        os.makedirs(base_dir_path)
    
    os.chdir(base_dir_path)

def download_stanford(download_url, filename='downloaded_data.zip'):
    """ Download and extract data """
    
    downloaded_filename = os.path.join('.', filename)
    print ('Downloading stanford data')
    urllib.request.urlretrieve(download_url,downloaded_filename)
    print ('Extracting stanford data')
    zipfile=ZipFile(downloaded_filename)
    zipfile.extractall('./')
    zipfile.close()

def download_twitter_data(download_url,filename = 'Tweets.csv'):
    downloaded_filename = os.path.join('.', filename)
    print ('Downloading Twitter data')
    urllib.request.urlretrieve(download_url,downloaded_filename)
    
    
    
def download_data(base_path):
    
    base_folder='data'
    
    # URL to download the sentiment140 dataset
    data_url1='http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
    data_url2 = 'https://raw.githubusercontent.com/sakibchowdhury131/Emotion_detection/master/final/data/Tweets.csv'
    
    base_dir_path=base_path+'/'+base_folder
    change_base_dir(base_dir_path)
    download_stanford(data_url1)
    download_twitter_data(data_url2)
