import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

train = pd.read_csv('data/train.csv').set_index('PassengerId')
test = pd.read_csv('data/test.csv').set_index('PassengerId')
output_cols = ['PassengerId', 'Survived']  # column headings for output file

'''
Variable	Definition	    Key
survival	Survival        0 = No, 1 = Yes
pclass	    Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	        Sex	
Age	        Age in years	
sibsp	    # of siblings / spouses aboard the Titanic	
parch	    # of parents / children aboard the Titanic	
ticket	    Ticket number	
fare	    Passenger fare	
cabin	    Cabin number	
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
'''

# view data
desc = train.describe()
'''
       PassengerId  Survived  Pclass     Age   SibSp   Parch    Fare
count       891.00    891.00  891.00  714.00  891.00  891.00  891.00
mean        446.00      0.38    2.31   29.70    0.52    0.38   32.20
std         257.35      0.49    0.84   14.53    1.10    0.81   49.69
min           1.00      0.00    1.00    0.42    0.00    0.00    0.00
25%         223.50      0.00    2.00   20.12    0.00    0.00    7.91
50%         446.00      0.00    3.00   28.00    0.00    0.00   14.45
75%         668.50      1.00    3.00   38.00    1.00    0.00   31.00
max         891.00      1.00    3.00   80.00    8.00    6.00  512.33
'''

N = train.shape[0]  # num samples
missing_samples = pd.DataFrame(data=[[train[col].isnull().sum(), train[col].isnull().sum()/N] for col in train.columns],
                               index=train.columns,
                               columns=['number_missing', 'proportion_missing']).round(3)  # missing samples per feature
'''
             number_missing  proportion_missing
PassengerId               0               0.000
Survived                  0               0.000
Pclass                    0               0.000
Name                      0               0.000
Sex                       0               0.000
Age                     177               0.199
SibSp                     0               0.000
Parch                     0               0.000
Ticket                    0               0.000
Fare                      0               0.000
Cabin                   687               0.771
Embarked                  2               0.002

1) Age, Cabin and Embarked are missing samples. 
2) Cabin is an unusable feature based on the number that are missing
3) Age might be able to be inferred or imputed, although 20% of values is a large number to impute
4) Embarked is categorical so can't be imputed -- ignore samples or ignore feature?
5) Ticket number is alphanumeric - unsure if there's anything in there. Ignore for now.
'''



'''
BASELINE

I'll start with a basic random forrest with:
1) Imputing the mean of Age in the missing values
2) Not using Cabin feature
3) Ignoring 2 missing samples in Embarked
4) Encoding categorical variables
'''

# data prep
def prep_baseline(data, test_or_train='train', imputer=None):
    """
    First go at preparing data for Random Forest
    :param data: DataFrame of data that is to be prepared for input to the model
    :param test_or_train: string either 'train' or 'test', indicating if it's training or testing data to be processed
    :param imputer: SimpleImputer fit to the training data
    :return: DataFrame of prepared data in the form ready to go into the model
    """

    # split data into features and labels
    if test_or_train == 'train':
        features = data.drop('Survived', axis=1)
        labels = data['Survived']
    elif test_or_train == 'test':
        features = data
        labels = None
    else:
        raise Exception(ValueError('value of test_or_train must be \'train\' or \'test\''))
    # Sex to numeric
    features['is_female'] = features['Sex'].map(dict(male=0, female=1))
    # ignore samples with missing values in Embarked
    mask = features['Embarked'].notnull()
    features = features[mask]
    if test_or_train == 'train':
        labels = labels[mask]
    # encode categorical columns
    categorical_columns = ['Embarked']
    features = pd.get_dummies(features, columns=categorical_columns)
    # remove unused columns
    drop_cols = ['Sex', 'Name', 'Ticket', 'Cabin']
    features.drop(drop_cols, axis=1, inplace=True)
    # impute means
    if imputer is None:
        imputer = SimpleImputer()
        imputer.fit(features)
    features = pd.DataFrame(data=imputer.transform(features), index=features.index, columns=features.columns)
    return [features, labels, imputer]

# prepare test and training data
X_train, y_train, imputer = prep_baseline(train)
X_test, _, _ = prep_baseline(test, test_or_train='test', imputer=imputer)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_test = pd.DataFrame(zip(X_test.index, model.predict(X_test)), columns=['PassengerId', 'Survived'])
y_test.to_csv('data/predictions_0.csv', index=False)
