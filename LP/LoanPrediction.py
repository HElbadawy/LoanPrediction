import csv
import statistics as s
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

modelLR = LogisticRegression()
modelSVM = svm.SVC()


def assign(reader):  # function returns evidence and label lists
    evidence = []
    label = []
    for row in reader:  # [yes, 0, graduate, yes] -> [1, 0, 1, 1]
        e = []
        columns = {2, 3, 4, 5, 8}
        for i in columns:  # checks if a row is empty
            if row[i] == '':
                row[i] = replace_list[i]

        # marriage
        e.append(int(row[2] == 'Yes'))

        # dependencies
        dependency = row[3]
        if dependency == '0':
            e.append(0)
        elif dependency == '1':
            e.append(1)
        elif dependency == '2':
            e.append(2)
        elif dependency == '3+':
            e.append(3)

        # education
        e.append(int(row[4] == 'Graduate'))

        # self_employed
        e.append(int(row[5] == 'Yes'))

        # loan_amount
        e.append(int(row[8]))

        evidence.append(e)

        # loan_status
        label.append(int(row[12] == 'Y'))
    return evidence, label


def replace_null():
    with open('train.csv') as trainStat:
        marriage = [row["Married"] for row in csv.DictReader(trainStat)]
    with open('train.csv') as trainStat:
        dependencies = [row["Dependents"] for row in csv.DictReader(trainStat)]
    with open('train.csv') as trainStat:
        education = [row["Education"] for row in csv.DictReader(trainStat)]
    with open('train.csv') as trainStat:
        employment = [row["Self_Employed"] for row in csv.DictReader(trainStat)]
    with open('train.csv') as trainStat:
        loanAmount = [row["LoanAmount"] for row in csv.DictReader(trainStat)]

    # removing null values in lists
    marriage = list(filter(None, marriage))
    dependencies = list(filter(None, dependencies))
    education = list(filter(None, education))
    employment = list(filter(None, employment))
    loanAmount = list(filter(None, loanAmount))
    loanAmount = list(map(int, loanAmount))

    # calculating mode/mean for lists
    marriageMode = s.mode(marriage)
    dependenciesMode = s.mode(dependencies)
    educationMode = s.mode(education)
    employmentMode = s.mode(employment)
    loanAmountMean = round(s.mean(loanAmount))

    # list to replace null values with(used in for loop above)
    lst = [None, None, marriageMode, dependenciesMode, educationMode, employmentMode, None, None, loanAmountMean]
    return lst


if __name__ == "__main__":
    replace_list = replace_null()

    with open('train.csv') as train:  # open csv file
        train_reader = csv.reader(train)  # convert csv file to list
        next(train_reader)  # skip the first row in csv
        train_evidence, train_label = assign(train_reader)

    X_train, X_test, y_train, y_test = train_test_split(train_evidence, train_label, test_size=0.3, random_state=0)

    modelLR.fit(X_train, y_train)
    modelSVM.fit(X_train, y_train)

    predictionLR = modelLR.predict(X_test)
    predictionSVM = modelSVM.predict(X_test)  

    print("Accuracy of Logistic Regression Classifier: ", metrics.accuracy_score(y_test, predictionLR))
    print("Classification Report of Logistic Regression Classifier:\n", classification_report(y_test, predictionLR))
    print("Confusion Matrix of Logistic Regression Classifier:\n", confusion_matrix(y_test, predictionLR))

    #print("Classification Report of SVM Classifier:\n", classification_report(y_test, predictionSVM))
    #print("Confusion Matrix of SVM Classifier:\n", confusion_matrix(y_test, predictionSVM))

    print("Accuracy of SVM Classifier: ", metrics.accuracy_score(y_test, predictionSVM))
    print("Precision of SVM Classifier: ", metrics.precision_score(y_test, predictionSVM))
    print("Recall of SVM Classifier: ", metrics.recall_score(y_test, predictionSVM))