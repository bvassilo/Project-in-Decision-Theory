import nltk                                                                     #natural language toolkit -work with human language data
import tensorflow as tf                                                         #to develop and train the model

import pandas as pd                                                             #fast data analysis 
import numpy as np                                                              #mathematic operations in python
import matplotlib.pyplot as plt                                                 #to plot data and results

from tqdm import tqdm                                                           #progress meters to visualize for loops and easier debug

from sklearn.feature_extraction.text import TfidfVectorizer                     #to tokenize our data

from nltk.tokenize import word_tokenize                                         #removes punctuations except from periods
from nltk.stem import PorterStemmer                                             #mapping a group of words to the same stem even if the stem itself is not a valid word in the Language
from nltk.corpus import stopwords                                               #removing stopwords such as a, the, this

from tensorflow.keras import Sequential                                         #for the model  
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization    #parameters to examine in our model    
from tensorflow.keras import regularizers                                       #parameters to examine in our model

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier              #classify data  

from sklearn.metrics import recall_score, f1_score, precision_score             #model metrics to calculate accuracy

df_neg = pd.read_csv('/contents/negative.csv')                                  #read the negative comments 
df_pos = pd.read_csv('/contents/positive.csv')                                  #read the positive comments

df_neg.head(10)                                                                 #print first ten to see if the import was correct
df_pos.head(10)

ds_neg = df_neg['Text'].values                                                  #obtain the comments and store them in a vector
ds_neg = np.stack([ds_neg, np.zeros_like(ds_neg)], axis=1)                      #stack another column with the label

ds_pos = df_pos['Text'].values
ds_pos = np.stack([ds_pos, np.ones_like(ds_pos)], axis=1)

                                                                               
ds_neg[0], ds_neg.shape, ds_pos[0], ds_neg.shape                                #array col 0 is text and col 1 is label (0 for negative, 1 for positive)

dataset = np.vstack([ds_neg, ds_pos])                                           #stack the negative and positive comments to one dataset
dataset.shape                                                                   

#"""# Preprocessing"""

nltk.download('punkt')                                                          #download punctuations                                                       
nltk.download('stopwords')                                                      #download stopwords

sentences = dataset[:, 0]                                                       #isolate all the comments
labels = dataset[:, 1].astype('float')                                          #isolate the labels

stemmer = PorterStemmer()                                                       #initialize the stemmer
stop_words = set(stopwords.words('english'))                                    #dictionary for stopwords in english

tokens = []                                                                     

print('Tokenizing, Stemming, Removing Stopwords')
print()
for i, sentence in enumerate(tqdm(sentences)):                                  
    tokens = word_tokenize(sentence)                                            #tokenize a sentence 
    stemmed = [stemmer.stem(x) for x in tokens]                                 #remove the suffix from the tokenized words
    t = [w for w in stemmed if w not in stop_words]                             #remove the stopwords

    sentences[i] = ' '.join(t)                                                  #join what is left
    tokens.append(t)

print()
print('Calculating Tfidf')
vectorizer = TfidfVectorizer()                                                  #Tfidf initialization
X = vectorizer.fit_transform(sentences).astype('float')                         #fit transform and then store as floats

train_test_limit = int(0.75 * X.shape[0])                                       #75% of the dataset is for training
X_test = X[train_test_limit:]                                                   #25% for testing
X = X[:train_test_limit]

y_test = labels[train_test_limit:]                                              #25% for testing
y = labels[:train_test_limit]                                                   #75% for training

#"""# Build and Train Model"""

def get_model():                                                                                          #model creation 
    wd = 1e-5
    model = Sequential()
    model.add(Dense(64, input_dim=X.shape[1], activation='relu', kernel_regularizer=regularizers.l2(wd),  #if we chose linear activation function in a multiple layer network it 
                activity_regularizer=regularizers.l2(wd)))                                                #can colapse to an equivalent network of 2 layers 
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(wd),                        #that is why we choos e relu which is a non linear activation function 
                activity_regularizer=regularizers.l2(wd)))
    model.add(Dense(1, activation='sigmoid'))                                                             #we chose sigmoid activation function because we need output of 0 or 1
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['binary_accuracy'])            #after experimentation adamax provided the best results

    return model

clf = KerasClassifier(build_fn=get_model)                                                                             #training the model
history = clf.fit(X.toarray(), y, validation_data=(X_test.toarray(), y_test), epochs=20, batch_size=32, shuffle=True) #definig model parameters, e:10 b:16, b:32 - e:20 b:16,b:32, with the parameters above e:20, b32 provides the best results

#"""# Result Visualization"""

# Plot training & validation binary accuracy values
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()