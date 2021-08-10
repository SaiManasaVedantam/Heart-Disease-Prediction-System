import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt; plt.rcdefaults()
import warnings
warnings.filterwarnings("ignore", category = Warning)

def classDivision(dataset):
    divided = {}
    for i in range(len(dataset)):
        row = dataset.iloc[i,:]
        if(row[-1] not in divided):
            divided[row[-1]] = []
        divided[row[-1]].append(row)
    return divided

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def classMeanStdDev(dataset):
    meanStdDev = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del(meanStdDev[-1])
    return meanStdDev

def classSummary(dataset):
    separated = classDivision(dataset)
    meanStdDev = {}
    for classValue, instances in separated.items():
        meanStdDev[classValue] = classMeanStdDev(instances)
    return meanStdDev

def findProb(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def findClassProb(meanStdDev, inputrow):
    probabilities = {}
    for classValue, classmeanStdDev in meanStdDev.items():
        probabilities[classValue] = 1
        for i in range(len(classmeanStdDev)):
            mean, stdev = classmeanStdDev[i]
            x = inputrow.iloc[i]
            probabilities[classValue] *= findProb(x, mean, stdev)
	return probabilities

def predict(meanStdDev, inputrow):
    probabilities = findClassProb(meanStdDev, inputrow)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(meanStdDev, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(meanStdDev, testSet.iloc[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 1
    for i in range(len(testSet)):
        if testSet.iloc[i,-1] == predictions[i]:
        	correct += 1
    return (correct/float(len(testSet))) * 100.0

class HDPS:
    
    def read_and_cleanData(self,filename):
        df=pd.read_csv(filename)
        df = df.apply(pd.to_numeric, errors='coerce')
        newDS=df
        newDS.fillna(0,inplace=True)
        newDS.to_csv('Processed.csv')
        df = pd.read_csv('Processed.csv')
        df = df.drop('Unnamed: 0', 1)
        return df
    
    def split_and_Scale(self,df):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1400)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        sets = [X_train,X_test,y_train,y_test]
        return sets
    
    
    def NaiveBayes(self,filename):
        #print('NAIVE BAYES')
        df = pd.read_csv(filename)
        df = df.apply(pd.to_numeric, errors='coerce')
        newDS = df
        newDS.fillna(0,inplace=True)
        newDS.to_csv('Processed.csv')
        data = pd.read_csv('Processed.csv')
        data = data.drop('Unnamed: 0', 1)
		
        trainSet,testSet = train_test_split(data, test_size = 0.20, random_state = 1400)
        meanStdDev = classSummary(trainSet)
        y_pred = getPredictions(meanStdDev, testSet)		
              
        # Making the Confusion Matrix
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1400)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_pred,y_test)
        accuracy = accuracy * 100
        
        pca = PCA(n_components=3)  
        X_train = pca.fit_transform(X_train)  
        X_test = pca.transform(X_test) 

        # Fitting Naive Bayes to the Training set
        #print('\nNEW NAIVE BAYES')
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        cm1 = confusion_matrix(y_test, y_pred)
        accuracy1 = accuracy_score(y_pred,y_test)
        accuracy1 = accuracy1 * 100
        return [cm,accuracy,cm1,accuracy1]
        
    def SVM(self,sets):    
        #print('\nSUPPORT row MACHINE')
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(sets[0], sets[2])
        
        # Predicting the Test set results
        y_pred = classifier.predict(sets[1])
        y_test = sets[3]
              
        # Making the Confusion Matrix
        cm = confusion_matrix(sets[3], y_pred)

        accuracy = accuracy_score(y_pred,sets[3])
        accuracy = accuracy * 100

        # Fitting SVM to the Training set
        #print('\nNEW SUPPORT row MACHINE')
        pca = PCA(n_components=4)  
        sets[0] = pca.fit_transform(sets[0])  
        sets[1] = pca.transform(sets[1]) 

        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(sets[0], sets[2])
        
        # Predicting the Test set results
        y_pred = classifier.predict(sets[1])
        y_test = sets[3]
              
        # Making the Confusion Matrix
        cm1 = confusion_matrix(sets[3], y_pred)

        accuracy1 = accuracy_score(y_pred,sets[3])
        accuracy1 = accuracy1 * 100
        return [cm,accuracy,cm1,accuracy1]
        
    def Logistic(self,sets):      
        #print('\nLOGISTIC REGRESSION')
        classifier = LogisticRegression()
        classifier.fit(sets[0], sets[2])
        
        # Predicting the Test set results
        y_pred = classifier.predict(sets[1])
        y_test = sets[3]
              
        # Making the Confusion Matrix
        cm = confusion_matrix(sets[3], y_pred)

        accuracy = accuracy_score(y_pred,sets[3])
        accuracy = accuracy * 100

        # Fitting SVM to the Training set
        #print('\nNEW LOGISTIC REGRESSION')
        pca = PCA(n_components=4)  
        sets[0] = pca.fit_transform(sets[0])  
        sets[1] = pca.transform(sets[1]) 

        classifier = LogisticRegression()
        classifier.fit(sets[0], sets[2])
        
        # Predicting the Test set results
        y_pred = classifier.predict(sets[1])
        y_test = sets[3]
              
        # Making the Confusion Matrix
        cm1 = confusion_matrix(sets[3], y_pred)

        accuracy1 = accuracy_score(y_pred,sets[3])
        accuracy1 = accuracy1 * 100
        return [cm,accuracy,cm1,accuracy1]
        
def full_proj(filename):
    ob1 = HDPS()
    nb_result=ob1.NaiveBayes(filename)
	
    ob2 = HDPS()
    data = ob2.read_and_cleanData(filename)
    result = ob2.split_and_Scale(data)
    log_result=ob2.Logistic(result)
	
    ob3 = HDPS()
    data = ob3.read_and_cleanData(filename)
    result = ob3.split_and_Scale(data)
    svm_result=ob3.SVM(result)
	
    objects = ('SVM',  'Naive Bayes', 'Logistic Regression')
    y_pos = np.arange(len(objects))
    performance = [svm_result[3],nb_result[3],log_result[3]]
    for color in ['r', 'b', 'g']:
        plt.plot(objects, performance, color=color)
		
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.savefig('static/Line')
    plt.clf()
    bar_width=0.25
    plt.bar(y_pos, performance,bar_width,align='center',alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.savefig('static/Bar')
    plt.clf()
    return [nb_result,log_result,svm_result]
	
def check(values,filename):
    name1 = "Heart1.csv"
    name2 = "Heart2.csv"
	
    if filename == name1:
		if values[0] < 0 or values[0] > 1:
			msg = "Invalid Value 1. Refer USER HELP for guidance"
		elif values[1] > 100 or values[1] < 10:
			msg = "Invalid Value 2. Refer USER HELP for guidance"
		elif values[2] < 1 or values[2] > 4:
			msg = "Invalid Value 3. Refer USER HELP for guidance"
		elif values[3] < 0 or values[3] > 1:
			msg = "Invalid Value 4. Refer USER HELP for guidance"
		elif (values[3] == 1 and values[4] == 0) or (values[3] == 0 and values[4] != 0):
			msg = "Invalid Value 5. If you are a current smoker, then give valid count of cigarettes you smoke per day"
		elif values[4] < 0: 
			msg = "Invalid Value 5. Refer USER HELP for guidance"
		elif values[5] < 0 or values[5] > 1:
			msg = "Invalid Value 6. Refer USER HELP for guidance"
		elif values[6] < 0 or values[6] > 1:
			msg = "Invalid Value 7. Refer USER HELP for guidance"
		elif values[7] < 0 or values[7] > 1:
			msg = "Invalid Value 8. Refer USER HELP for guidance"
		elif values[8] < 0 or values[8] > 1:
			msg = "Invalid Value 9. Refer USER HELP for guidance"
		elif values[9] < 80 or values[9] > 450:
			msg = "Invalid Value 10. Refer USER HELP for guidance"
		elif values[10] < 90 or values[10] > 250:
			msg = "Invalid Value 11. Refer USER HELP for guidance"
		elif values[11] < 60 or values[11] > 140:
			msg = "Invalid Value 12. Refer USER HELP for guidance"
		elif values[12] < 18.5 or values[12] > 30:
			msg = "Invalid Value 13. Refer USER HELP for guidance"
		elif values[13] < 50 or values[13] > 160:
			msg = "Invalid Value 14. Refer USER HELP for guidance"
		elif values[14] < 50 or values[14] > 130:
			msg = "Invalid Value 15. Refer USER HELP for guidance"
		else:
			msg = checkRecord(values,filename)
		return msg
		
    if filename == name2:
		if values[0] < 10 or values[0] > 100:
			msg = "Invalid Value 1. Refer USER HELP for guidance"
		elif values[1] < 0 or values[1] > 1:
			msg = "Invalid Value 2. Refer USER HELP for guidance"
		elif values[2] < 0 or values[2] > 1:
			msg = "Invalid Value 3. Refer USER HELP for guidance"
		elif values[3] < 1 or values[3] > 4:
			msg = "Invalid Value 4. Refer USER HELP for guidance"
		elif values[4] < 0 or values[4] > 1: 
			msg = "Invalid Value 5. Refer USER HELP for guidance"
		elif values[5] < 20 or values[5] > 80:
			msg = "Invalid Value 6. Refer USER HELP for guidance"
		elif values[6] < 18.5 or values[6] > 30:
			msg = "Invalid Value 7. Refer USER HELP for guidance"
		elif values[7] < 1.0 or values[7] > 6.0:
			msg = "Invalid Value 8. Refer USER HELP for guidance"
		elif values[8] < 0 or values[8] > 1:
			msg = "Invalid Value 9. Refer USER HELP for guidance"
		elif values[9] < 100 or values[9] > 450:
			msg = "Invalid Value 10. Refer USER HELP for guidance"
		elif values[10] < 90 or values[10] > 250:
			msg = "Invalid Value 11. Refer USER HELP for guidance"
		elif values[11] < 60 or values[11] > 140:
			msg = "Invalid Value 12. Refer USER HELP for guidance"
		elif values[12] < 50 or values[12] > 160:
			msg = "Invalid Value 13. Refer USER HELP for guidance"
		elif values[13] < 50 or values[13] > 130:
			msg = "Invalid Value 14. Refer USER HELP for guidance"
		elif values[14] < 18.5 or values[14] > 30:
			msg = "Invalid Value 15. Refer USER HELP for guidance"
		else:	
			msg = checkRecord(values,filename)
		return msg
		
def checkRecord(values,filename):
	ob = HDPS()
	data = ob.read_and_cleanData(filename)
	result = ob.split_and_Scale(data)
	
	#result[0] = np.vstack((result[0],result[1]))
	#result[2] = np.hstack((result[2],result[3]))
		
	X_test = result[1]
	X_test = X_test[np.isnan(X_test).any(axis=1)]
	X_test = np.vstack((X_test,values))
	
	gnb = GaussianNB()
	gnb.fit(result[0], result[2])
	gnb_pred = gnb.predict(X_test)
	
	svm = SVC(kernel = 'poly', random_state = 0)
	svm.fit(result[0], result[2])
	svm_pred = svm.predict(X_test)
		
	logr = LogisticRegression()
	logr.fit(result[0], result[2])
	logr_pred = logr.predict(X_test)

	sum = logr_pred + (0.9 * svm_pred) + (0.8 * gnb_pred)
	
	print gnb_pred
	print svm_pred
	print logr_pred
	print sum
	
	if sum == 0:
		res="Relax! Your Heart is Healthy. Keep it up!"
	else:
		res="Take Care! You may get attacked by Heart Disease in near future"
	return res	