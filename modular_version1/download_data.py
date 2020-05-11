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
    if not os.path.exists(downloaded_filename):
        print ('Downloading stanford data')
        urllib.request.urlretrieve(download_url,downloaded_filename)
    else:
        print('Stanford dataset already downloaded...')
        
    if not os.path.exists('training.1600000.processed.noemoticon.csv'):
        print ('Extracting stanford data')
        zipfile=ZipFile(downloaded_filename)
        zipfile.extractall('./')
        zipfile.close()

def download_twitter_data(download_url,filename = 'Tweets.csv'):
    downloaded_filename = os.path.join('.', filename)
    if not os.path.exists(downloaded_filename):
        print ('Downloading Twitter data')
        urllib.request.urlretrieve(download_url,downloaded_filename)
    else:
        print('Twitter dataset already downloaded...')


def download_glove_model(download_url, filename='glove.6B.zip'):
    """ Download and extract data """
    
    downloaded_filename = os.path.join('.', filename)
    if not os.path.exists(downloaded_filename):
        print ('Downloading glove model...')
        urllib.request.urlretrieve(download_url,downloaded_filename)
    else:
        print('Glove model already downloaded...')
    if not os.path.exists('glove.6B.100d.txt'):
        print ('Extracting glove data')
        zipfile=ZipFile(downloaded_filename)
        zipfile.extractall('./')
        zipfile.close()

    
    
    
def extract_file(file_path):
    print ('Extracting file')
    zipfile=ZipFile(file_path)
    zipfile.extractall('./')
    zipfile.close()
    
def download_data(base_path):
    
    base_folder='data'
    
    # URL to download the sentiment140 dataset
    data_url1='http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
    data_url2 = 'https://raw.githubusercontent.com/sakibchowdhury131/Emotion_detection/master/final/data/Tweets.csv'
    glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    
    base_dir_path=base_path+'/'+base_folder
    change_base_dir(base_dir_path)
    download_stanford(data_url1)
    download_twitter_data(data_url2)
    download_glove_model(glove_url)
    change_base_dir(base_path)
