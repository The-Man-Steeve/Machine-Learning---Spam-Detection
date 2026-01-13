import nltk
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords') #gets the stopwords
nltk.download('punkt') #gets the punctuation
nltk.download('punkt_tab') #im not sure why this is needed but my IDE won't work without it
s_words = set(stopwords.words('english'))


#GET THE INPUT
print('***Email Extraction Program***')
dataset_no = 0
while not (dataset_no == 1 or dataset_no == 2 or dataset_no == 4):
    dataset_no = int(input('Select dataset (enter 1, 2, 4): '))
print('Input recieved!')
#-----------------------------------------------------------------------------------------------------
folders = ['dataset\\' + 'enron' + str(dataset_no) + '_train\\ham',
           'dataset\\' + 'enron' + str(dataset_no) + '_train\\spam',
           'dataset\\' + 'enron' + str(dataset_no) + '_test\\ham',
           'dataset\\' + 'enron' + str(dataset_no) + '_test\\spam'
           ]
#-----------------------------------------------------------------------------------------------------
print('Extracting Data...')

master_matrix = []
vocabulary_set = set() #global set of vocabulary
words = [] #this is the global header row
vocab_words_hash = {} #this maps a word to its index in the global header row
vocab_size = 0 #vocab_size = number of columns
num_rows = 0 #number of rows = number of files
cutoff = 0 #catches the number of ham files

for i in range(2):
    entries = os.scandir(folders[i])
    for entry in entries:
        num_rows += 1 #each row represents a different file
        row_vocabulary = {} #local vocabulary
        if entry.is_file():
            open_file = open(folders[i] + '\\' + entry.name, mode='r', encoding='latin1')
            fileText = open_file.read() 
            tokens = word_tokenize(fileText)
            for item in tokens:
            #set the string to all lower-case
                item = item.lower()
                #we check to verify if it is a valid feature
                #ignore numbers, punctuation, words of length 1, and stopwords
                if item.isalpha() and len(item) > 2 and  item not in s_words:
                    if item not in row_vocabulary: #if we haven't seen the word in this email
                        row_vocabulary[item] = 1
                    else:
                        row_vocabulary[item] += 1
                    if item not in vocabulary_set: #if we have never seen the word ever
                        vocabulary_set.add(item)
                        words.append(item)
                        vocab_words_hash[item] = vocab_size
                        vocab_size += 1
            master_matrix.append(row_vocabulary)
            open_file.close()
    if i == 0:
        cutoff = num_rows
words.append('label')

print('Complete!')
#print(len(vocabulary_set))

print('Generating Training Matrices...')
BOW_train = np.zeros((num_rows, (vocab_size) + 1), dtype=int)
Bernoulli_train = np.zeros((num_rows, (vocab_size) + 1),dtype=int)

#print(BOW.shape)
#print(Bernoulli.shape)
#print(vocab_words_hash['subject'])


#now we populate the Bag of Words and Bernoulli matrices
r = 0 #iterator
for s in master_matrix:
    for key, value in s.items(): 
        column_index = vocab_words_hash[key]
        BOW_train[r, column_index] += value
        Bernoulli_train[r,column_index] = 1
        if r < cutoff:
            BOW_train[r, -1] = 0
            Bernoulli_train[r, -1] = 0
        else:
            BOW_train[r, -1] = 1
            Bernoulli_train[r, -1] = 1
    r += 1
print('Complete!')
'''
verification
#print(Bernoulli[318,-1])
#print(Bernoulli[319, -1])
print(words[5])
print(vocab_words_hash['please'])
print(BOW[0,5])
print(Bernoulli[0,5])
'''
#SAVE THE TRAINING FILES
print('Saving training data...')
#----------------------------------------------------------------------------------------------------------------
np.savetxt('enron' + str(dataset_no) + '_bow_train.csv', BOW_train, delimiter=',', header=','.join(words), fmt= "%d")
np.savetxt('enron' + str(dataset_no) + '_bernoulli_train.csv', Bernoulli_train, delimiter=',', header=','.join(words), fmt= "%d")
#----------------------------------------------------------------------------------------------------------------
print('Complete!')

#i need to calculate the number of rows in the test data
#there is a more efficient way to do this, but I don't care
#it works
test_rows = 0
for i in range(2,4):
    f = os.listdir(folders[i])
    test_rows += len(f)
#print(test_rows)

print('Generating the testing Matrices...')
BOW_test = np.zeros((test_rows, vocab_size + 1), dtype=int)
Bernoulli_test = np.zeros((test_rows, vocab_size + 1), dtype=int)
#now for the test Set

r = 0
for i in range(2,4):
    entries = os.scandir(folders[i])
    for entry in entries:
        #print(entry.name)
        if entry.is_file():
            open_file = open(folders[i] + '\\' + entry.name, mode='r', encoding='latin1')
            #print("decoding r#", r)
            fileText = open_file.read()
            tokens = word_tokenize(fileText)
            open_file.close()
            for item in tokens:
                item = item.lower()
                if item in vocabulary_set:
                    #print(item)
                    BOW_test[r,vocab_words_hash[item]] += 1
                    Bernoulli_test[r, vocab_words_hash[item]] = 1
        if i == 2:
            BOW_test[r, -1] = 0 #setting the class label
            Bernoulli_test[r, -1] = 0
        else:
            BOW_test[r, -1] = 1 #setting the class label
            Bernoulli_test[r, -1] = 1
        open_file.close()
        r += 1
'''
debugging
print(vocab_words_hash['hou'])
print(BOW_test[0,vocab_words_hash['hou']])
print(Bernoulli_test[0,vocab_words_hash['hou']])
'''
print('Complete!')
#SAVE THE TEST FILES
print('Saving Training Data...')
#--------------------------------------------------------------------------------------------------------------
np.savetxt('enron' + str(dataset_no) + '_bernoulli_test.csv', Bernoulli_test, delimiter=',', header=','.join(words), fmt= "%d")
np.savetxt('enron' + str(dataset_no) + '_bow_test.csv', BOW_test, delimiter=',', header=','.join(words), fmt= "%d")
#--------------------------------------------------------------------------------------------------------------
print('Complete!')


print('Have a nice Day! :)')