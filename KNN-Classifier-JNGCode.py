import numpy as np
from joblib import dump, load
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

def main():
    digits = datasets.load_digits() #loading the dataset from sklearn of handwritten digits
    k = 5 #k group nearest number
    x = digits.data
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) #split data training data 80:20 testing data
#menu option for the user
    print('Menu: ')
    print('1. Database Information')
    print('2. KNN SKLearn Algorithm with accuracy and error result')
    print('3. My KNN Algorithm with accuracy and error result')
    print('4. Exit')
    user = input('Select an option, 1, 2, 3 or 4: ')
    if user == '1':
        dataset_info(digits, x, y)
    elif user == '2':
        skl_model(x_train, x_test, y_train, y_test)
    elif user == '3':
        myKNN(x_train, y_train, x_test, y_test, k)
    elif user == '4':
        exit()
    else:
        print('ERROR MESSAGE: invalid input...')
        main()

def dataset_info(digits, x, y):
    MinMax = [] #minimum = MinMax[0], maximum = MinMax[1] - respectively.
    for inArray in x:
        MinMax.append(min(inArray)) #minimum found
        MinMax.append(max(inArray)) #maximum found
    print('------------------------------------------------------------------')
    print('[F1]: Dataset information: ') 
    print('Number of keys:', digits.keys(), '||', len(digits.keys()), '- keys')
    print('Data number of entries:', digits.target_names, '||', len(digits.target_names), '- Total entries')
    print('Data shape:', digits.data.shape, '|| (64 - matrix)')
    print('Minimum value: ', min(x[0]))
    print('Maximum value: ', max(x[1]))
    print('------------------------------------------------------------------')
    return main()

def skl_model(x_train, x_test, y_train, y_test):
    knnCLF = KNeighborsClassifier(n_neighbors = 5).fit(x_train, y_train) #calling sklearn classifier to find neigbors
    knnPredTest = knnCLF.predict(x_test) #using knn from sklearn to predict
    knnPredTestAcc = accuracy_score(y_test, knnPredTest) # Accuracy of testing data sklmodel
    knnPredTrain = knnCLF.predict(x_train)
    knnPredTrainAcc = accuracy_score(y_train, knnPredTrain) #Accuracy of the training data sklmodel
    print('------------------------------------------------------------------') 
    print('[F2]: SKL Model Results: ') #printing sklearn all results
    print('Error for SKLearn model of training set: {}'.format(1 - knnPredTrainAcc)) # [F4]Error of failure for training data.
    print('Accuracy Score Training Set: ', accuracy_score(y_train, knnPredTrain)) #Accuracy of Training Data
    print()
    print('Error for SKLearn model of testing set: {}'.format(1 - knnPredTestAcc)) #[F4] Error statistic for Testing Set via SKLearn
    print('Accuracy Score Testing Set: ', accuracy_score(y_test, knnPredTest)) #Accuracy statistic for Testing set Via SKLearn
    print('------------------------------------------------------------------')  
    
    dump(knnCLF, 'skl-model.joblib') #save the sklmodel to be called later.
    sklSaved = load('skl-model.joblib') #option to load sklmodel
    return sklSaved

def myKNN(x_train, y_train, x_test, y_test, k):
    allpredictions = [] #storing all predicted distances.
    for i in range(len(x_test)): #predicting the numeber closest to a class
        allpredictions.append(prediction(x_train, y_train, x_test[i, :], k))

    KNNaccuracy = accuracy_score(y_test, allpredictions) #knn accuracy of the score to the data.
    print('------------------------------------------------------------------')
    print('[F3]: My KNN Model: ')
    print('Error of my KNN Model: {}'.format(1 - KNNaccuracy)) #Print and calculate Error - [F4] task.
    print('Accuracy score of my KNN model: {}'.format(KNNaccuracy))
    print('------------------------------------------------------------------')
    dump(myKNN, 'myKNN-model.joblib') #save the sklmodel to be called later.
    myKNNSaved = load('myKNN-model.joblib') #option to load sklmodel
    main()

def prediction(x_train, y_train, x_test, k): #prediction via neighbors and euclidean distance
    ed = [] #Initialise variable arrays for Euclidean distance and target.
    target = []
    for i in range(len(x_train)):
        ed.append([np.sqrt(np.sum(np.square(x_test - x_train[i, :]))), i]) # sqrt all values in the array and calculating the euclidean distance and adding it to the array
    ed = sorted(ed)
    for i in range(k): #getting the neighbors
        element = ed[i][1]
        target.append(y_train[element])
    vote = Counter(target).most_common(1)[0][0]
    return vote

main()
