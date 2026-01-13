import numpy as np
import math
#-------------------------------------------------------------------------------
#STEP 1: EXTRACT THE DATA FROM THE CSV
#use these for the data files

#'enron1_bow_train.csv'
#'enron1_bow_test.csv'

#'enron2_bow_train.csv'
#'enron2_bow_test.csv'

#'enron4_bow_train.csv'
#'enron4_bow_test.csv'
#GET THE INPUT
print('***Multinomial Naive Bayes model***')
dataset_no = 0
while not (dataset_no == 1 or dataset_no == 2 or dataset_no == 4):
    dataset_no = int(input('Select dataset (enter 1, 2, 4): '))
#constructing the string of the files
training_file = 'enron' + str(dataset_no) + '_bow_train.csv'
test_file = 'enron' + str(dataset_no) + '_bow_test.csv'
#lines 19-21 are AI generated
with open(training_file, 'r') as f:
    header = f.readline().strip().split(',')
array = np.loadtxt(training_file,delimiter=',')
test_array = np.loadtxt(test_file, delimiter=',')
vocab_size = len(header) - 1
#print(header)
#print(array)
#-----------------------------------------------------------------------------
#STEP 2: COUNT THE FREQUENCY OF EACH CLASS TO GET THE PRIORS
num_spam = 0
num_ham = 0
for c in array[:,-1]:
    if c == 0:
        num_ham += 1
    else:
        num_spam += 1

prior_spam = num_spam / (num_spam + num_ham)
prior_ham = 1 - prior_spam
#print('total ham: ', num_ham)
#print('total spam:', num_spam)
#print('prior ham : ', prior_ham)
#print('prior spam: ', prior_spam)
#--------------------------------------------------------------------------------
#STEP 3: COUNT THE NUMBER OF INSTANCES OF EACH WORD FOR EACH CLASS
spam_class_count = np.zeros((1,vocab_size), dtype=int)
total_spam_words = 0
ham_class_count = np.zeros((1, vocab_size), dtype=int)
total_ham_words = 0
#each index in the above arrays corresponds directly with the word in its respective header index
#i.e. header[0] = 'subject', spam_class_count[0,0] = count('subject')
for j in range(vocab_size):
    for i in range(len(array)):
        ct = array[i, j]
        #print(array[i, 1], " -> ", array[i,-1])
        if array[i, -1] == 1: #if the class is spam
            total_spam_words += ct
            spam_class_count[0,j] += ct
        else: #the class is not spam
            total_ham_words += ct
            ham_class_count[0, j] += ct
#print(total_ham_words)
#print(total_spam_words)
#print(spam_class_count)
#print(ham_class_count)
#--------------------------------------------------------------
#STEP 4: CALCULATE THE CONDITIONAL PROBABILITY FOR EACH WORD
cp = np.zeros((2, vocab_size)) #holds the conditional probabilities
#at row 0, we look at class 0 (ham)
#at row 1, we look at class 1 (spam)
#APPLY LAPLACE SMOOTHING FOR THE DENOMINATORS
total_ham_words += vocab_size
total_spam_words += vocab_size 
for i in range(2):
    for j in range(vocab_size):
        #APPLY LAPLACE SMOOTHING FOR THE NUMERATORS
        if i == 0:
            cp[i,j] = (ham_class_count[0,j] + 1) / (total_ham_words)
        elif i == 1:
            cp[i,j] = (spam_class_count[0,j] + 1) / (total_spam_words)
'''
print('word = ', header[1])
print('ham')        
print(ham_class_count[0,1])
print(total_ham_words)
print(vocab_size)
print(cp[0,1])
print("spam")
print(spam_class_count[0,1])
print(total_spam_words)
print(vocab_size)
print(cp[1,1])
'''
#-------------------------------------------------------------------------------------
#STEP 5 CALCULATE THE PROBABILITIES THAT A TEST DOCUMENT IS SPAM/HAM
predictions = []
#we only have to calculate this once
prior_ham = math.log(prior_ham)
prior_spam = math.log(prior_spam)
for doc in range(len(test_array)):
    ham_probability = prior_ham
    spam_probability = prior_spam
    for i in range(vocab_size):
        ham_probability += test_array[doc,i] * math.log(cp[0,i])
        spam_probability += test_array[doc,i] * math.log(cp[1,i])
    #print(ham_probability)
    #print(spam_probability)
    if ham_probability > spam_probability:
        #print("THIS IS HAM")
        p = 0
        predictions.append(p)
    else:
        #print('THIS IS SPAM')
        p = 1
        predictions.append(p)
#--------------------------------------------------------------------------------
#STEP 6: MEASURE FOR ACCURACY, PRECISION, RECALL, AND F1
tp = 0
tn = 0
fp = 0
fn = 0
for k in range(len(predictions)):
    if predictions[k] == test_array[k, -1]: #if prediction is correct
        if predictions[k] == 1: #and prediction is 1 (positive)
            tp += 1
        else: #prediction is 0
            tn += 1
    else: #if prediction is inaccurate
        if predictions[k] == 1: #prediction is 1 (false positive)
            fp += 1
        else: #false negative
            fn += 1
#print(tp)
#print(tn)
#print(fp)
#print(fn)

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = ((precision * recall) / (precision + recall)) * 2

print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1_score)

