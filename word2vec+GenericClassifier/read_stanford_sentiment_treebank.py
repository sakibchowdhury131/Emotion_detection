import pandas as pd

def read_data (path) : 
    temp_data = pd.read_table (path+'/dictionary.txt')
    temp_data_processed = temp_data['!|0'].str.split('|',expand = True )
    temp_data_processed = temp_data_processed.rename(columns = {0:'Phrase', 1: 'phrase_ids'})
    
    temp_sentiment = pd.read_table(path+'/sentiment_labels.txt')
    temp_sentiment_processed = temp_sentiment['phrase ids|sentiment values'].str.split('|',expand = True)
    temp_sentiment_processed = temp_sentiment_processed.rename(columns = {0:'phrase_ids',1:'sentiment_values'})
    
    processed = pd.merge(temp_data_processed,temp_sentiment_processed, on = 'phrase_ids', how = 'left')
    return processed





