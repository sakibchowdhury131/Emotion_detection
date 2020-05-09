#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 22:11:52 2020

@author: sakib
"""
import pandas as pd
import re
import num2words
from nltk.tokenize import TweetTokenizer



def load_data_embedding(csv_file):
    df=pd.read_csv(csv_file,header = None, encoding='iso-8859-1') 
    df.columns=['Label','TweetId','Date','Query','User','Text']
    return df[['Text', 'Label']]
    


def load_data(csv_file): 
    dataset = pd.read_csv(csv_file)
    dataset = dataset.rename(columns={'airline_sentiment': 'Label', 'text': 'Text'})
    return dataset[['Text', 'Label']]


def preprocess_data(dataset):   
    """Read the raw tweet data from a file. Replace Emails etc with special tokens"""
    
        
    
    pos_emoticons=["(^.^)","(^-^)","(^_^)","(^_~)","(^3^)","(^o^)","(~_^)","*)",":)",":*",":-*",":]",":^)",":}",
                   ":>",":3",":b",":-b",":c)",":D",":-D",":O",":-O",":o)",":p",":-p",":P",":-P",":Þ",":-Þ",":X",
                   ":-X",";)",";-)",";]",";D","^)","^.~","_)m"," ~.^","<=8","<3","<333","=)","=///=","=]","=^_^=",
                   "=<_<=","=>.<="," =>.>="," =3","=D","=p","0-0","0w0","8D","8O","B)","C:","d'-'","d(>w<)b",":-)",
                   "d^_^b","qB-)","X3","xD","XD","XP","ʘ‿ʘ","❤","💜","💚","💕","💙","💛","💓","💝","💖","💞",
                   "💘","💗","😗","😘","😙","😚","😻","😀","😁","😃","☺","😄","😆","😇","😉","😊","😋","😍",
                   "😎","😏","😛","😜","😝","😮","😸","😹","😺","😻","😼","👍"]
    
    neg_emoticons=["--!--","(,_,)","(-.-)","(._.)","(;.;)9","(>.<)","(>_<)","(>_>)","(¬_¬)","(X_X)",":&",":(",":'(",
                   ":-(",":-/",":-@[1]",":[",":\\",":{",":<",":-9",":c",":S",";(",";*(",";_;","^>_>^","^o)","_|_",
                   "`_´","</3","<=3","=/","=\\",">:(",">:-(","💔","☹️","😌","😒","😓","😔","😕","😖","😞","😟",
                   "😠","😡","😢","😣","😤","😥","😦","😧","😨","😩","😪","😫","😬","😭","😯","😰","😱","😲",
                   "😳","😴","😷","😾","😿","🙀","💀","👎"]
    
    # Emails
    emailsRegex=re.compile(r'[\w\.-]+@[\w\.-]+')
    
    # Mentions
    userMentionsRegex=re.compile(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)')
    
    #Urls
    urlsRegex=re.compile('r(f|ht)(tp)(s?)(://)(.*)[.|/][^ ]+') # It may not be handling all the cases like t.co without http
    
    #Numerics
    numsRegex=re.compile(r"\b\d+\b")
    
    punctuationNotEmoticonsRegex=re.compile(r'(?<=\w)[^\s\w](?![^\s\w])')
    
    emoticonsDict = {}
    for i,each in enumerate(pos_emoticons):
        emoticonsDict[each]=' POS_EMOTICON_'+num2words.num2words(i).upper()+' '
        
    for i,each in enumerate(neg_emoticons):
        emoticonsDict[each]=' NEG_EMOTICON_'+num2words.num2words(i).upper()+' '
        
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in emoticonsDict.items())
    emoticonsPattern = re.compile("|".join(rep.keys()))
    

    
    
    all_lines = dataset['Text']
    padded_lines=[]
    for line in all_lines:
                line = emoticonsPattern.sub(lambda m: rep[re.escape(m.group(0))], line.lower().strip())
                line = userMentionsRegex.sub(' USER ', line )
                line = emailsRegex.sub(' EMAIL ', line )
                line=urlsRegex.sub(' URL ', line)
                line=numsRegex.sub(' NUM ',line)
                line=punctuationNotEmoticonsRegex.sub(' PUN ',line)
                line=re.sub(r'(.)\1{2,}', r'\1\1',line)
                words_tokens=[token for token in TweetTokenizer().tokenize(line)]                   
                line= ' '.join(token for token in words_tokens ) 
                line = line.lower()
                padded_lines.append(line)
    return padded_lines



def read_labels(dataset):
    labels = pd.to_numeric(dataset['Label'], downcast = 'float')
    labels = (labels > 1).astype(float)
    return labels

