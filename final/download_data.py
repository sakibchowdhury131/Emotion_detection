# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:34:44 2020

@author: sakib

"""

import os
from zipfile import ZipFile
import urllib.request



def download_data(download_url='http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip', filename='downloaded_data.zip',directory):
    """ Download and extract data """
    
    downloaded_filename = os.path.join(directory, filename)
    print ('Step 1: Downloading data')
    urllib.request.urlretrieve(download_url,downloaded_filename)
    print ('Step 2: Extracting data')
    zipfile=ZipFile(downloaded_filename)
    zipfile.extractall('./')
    zipfile.close()
