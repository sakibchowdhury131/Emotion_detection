import re



# filters texts      

def filter_texts (dataset,text_column_name = 'texts',things_to_keep = '[^a-zA-Z0-9\s@]'):      
    for i in range(0, dataset.shape[0]):
        comment = re.sub(things_to_keep, ' ', dataset[text_column_name][i])
        comment = comment.lower()
        #comment = comment.split()
        #comment = [j for j in comment if len(j) > 1]
        #comment = ' '.join(word for word in comment if not word.startswith('@'))
        dataset[text_column_name][i] = comment
    return dataset


for line in lines:
    tokens = word_tokenize(line)
    token = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    review_lines.append(words)

