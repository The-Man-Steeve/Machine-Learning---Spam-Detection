#STEPHEN BELCHER
#logistic.py
#---------------------------------------------------------
#IMPORTS
import numpy as np
from math import sqrt
#----------------------------------------------------------
#DEFINE A FUNCTION TO CREATE A PREDICTION of the documents
def predict(current_set, w):
    num_rows = len(current_set)
    #print('predicting values')
    y_pred = np.zeros(shape=(1,num_rows))
    z = np.zeros(shape=(1, num_rows))
    for i in range(num_rows):
        for j in range(0, n):
            z[0,i] = z[0,i] + (w[j] * current_set[i,j])
        #print('row ', i)
        #print('z = ', z[0,i])
        y_pred[0,i] = 1 / (1 + np.exp(-z[0,i]))
    #print(y_pred)
    return y_pred
#
#DEFINE A FUNCTION TO COMPUTE THE GRADIENT AND APPLY L2 REGULARIZATION
#INPUTS:
#y_pred = class predictions
#w = current set of weights
#reg = regularization constant
#stopflag: we send a reference to the stopflag so it can be altered outside of the function
def compute_gradient(y_pred, w, reg, stopFlag):
    norm = 0
    #print('computing gradient')
    g = np.zeros((1, n+1))
    for j in range(n):
        for i in range(1, len(y_pred[0])):
            g[0,j] = g[0,j] + (X[i,j] * (X[i,-1] - y_pred[0,i]))
        if(not(j == 0)):
            g[0,j] = g[0,j] - (reg * w[j])
            norm += g[0,j]**2
    #if the euclidian distance of the gradient is too small, then we will halt the process
    norm = sqrt(norm)
    if norm < 0.000001:
        stopFlag = True
    return (g,stopFlag) 
#
#TEST THE CURRENT MODEL
def test_the_model(current_set, w):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    p = predict(current_set, w)
    #print(p)

    for i in range(len(p[0])):
        if (p[0,i] >= 0.5):
            if current_set[i,-1] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if current_set[i, -1] == 0:
                tn += 1
            else:
                fn += 1
    #print('True positives: ', tp)
    #print('True negatives: ', tn)
    #print('False positives: ', fp)
    #print('False negatives: ', fn)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if precision + recall > 0:
        f1_score = ((precision * recall) / (precision + recall)) * 2
    else:
        f1_score = 0

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1_score)
    return [accuracy, precision, recall, f1_score]
#-----------------------------------------------------------------
'''
MAIN
'''
#GET THE INPUT
print('***Logistic Regression model***')
dataset_no = 0
while not (dataset_no == 1 or dataset_no == 2 or dataset_no == 4):
    dataset_no = int(input('Select dataset (enter 1, 2, 4): '))
counting_type = ''
while not(counting_type == 'bow' or counting_type == 'bernoulli'):
    counting_type = input('Select counting type (enter \'bow\' or \'bernoulli\'): ')
print('Input recieved!')


#PREPARE THE DATA AND INITIALIZE THE VARIABLES
training_file = 'enron' + str(dataset_no) + '_' + counting_type + '_train.csv'
#print('enron' + str(dataset_no) + '_' + counting_type + '_train.csv')
test_file = 'enron' + str(dataset_no) + '_' + counting_type + '_test.csv'
X = np.loadtxt(training_file, delimiter=',', dtype=int)
#print(X.shape)
#print(X[0,-1])
d = len(X) #total number of rows
n = len(X[0]) #number of features (plus 1 for bias feature, minus 1 for class label)
#these next two lines are AI generated
ones = np.ones((X.shape[0], 1))
X = np.hstack((ones, X))


#SPLIT THE DATA INTO TRAINING AND VALIDATION SETS
print('splitting data into training and validation sets...')
np.random.shuffle(X)
split = int(d * .7)
training_set = X[0:split, :]
validation_set = X[split:, :]
#print(X[:,0])
#print(len(X[0]))
#-------------------------------------------------------------
#HYPERPARAMETERS
learning_rate = .1
max_iterations = 125
#regularization_c = 1
#i didnt know exactly where to start when tuning the regularization constant,
#so I followed the advice of AI
reg_candidates = [.1,1,10,25]
validation_accuracy = [0,0,0,0]
#---------------------------------------------------------------
final_weights = np.random.uniform(low=-1, high=1, size=n) #initialize it just so we have the data structure
most_accurate_model = [0,0] #first element is the index of the regularization constant, second = accuracy
#TRAIN MODELS USING DIFFERENT REGULARIZATION CONSTANTS
for i in range(len(reg_candidates)):
    stopFlag = False
    regularization_c = reg_candidates[i]
    print('Training a model with regularization constant lambda = ', str(regularization_c))
    #initializing weights, this was with the help of AI
    #remember that the class label is the last element, so we dont include n+1
    weights = np.random.uniform(low=-.001, high= .001, size=n)
    #print(weights[0:5])
    #-------------------------------------------------------------------------------
    #GRADIENT ASCENT ALGORITHM
    print('Initiating Gradient Ascent with max = ' + str(max_iterations) + ' iterations')
    for t in range(max_iterations):
        print('iteration', t + 1)
        #print('predicting y values')
        y_predictions = predict(training_set, weights)
        #print('computing gradient')
        gradient,stopFlag = compute_gradient(y_predictions, weights, reg_candidates[i], stopFlag)
        #print(y_predictions)
        #print(gradient)
        #print('updating weights')
        for j in range(len(weights)):
            weights[j] = weights[j] + (learning_rate * gradient[0,j])
        #the stopflag is to indicate convergence, if we hit convergence, then stop
        if stopFlag == True:
            break

    print('Testing on the Validation Set:' )
    metrics = test_the_model(validation_set, weights)
    if metrics[0] > most_accurate_model[1]: #if we get a more accurate model
        final_weights = weights
        most_accurate_model[0] = i
        most_accurate_model[1] = metrics[0]

print('Testing on test_set...')
test_array = np.loadtxt(training_file, delimiter=',', dtype=int)
test_metrics = test_the_model(test_array, final_weights)
#print('Regularization Constant (Lambda) = ', str(regularization_c))
print('Test set = ', dataset_no, ' Type = ', counting_type)
print('regularization constant used: ', reg_candidates[most_accurate_model[0]])