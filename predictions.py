import pandas as pd

train = pd.read_csv('data/train.csv')

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
            PassengerId	Survived	Pclass	Age	    SibSp	Parch	 Fare
count	    891.0	    891.0	    891.0	714.0	891.0	891.0	 891.0
mean	    446.0	    0.38    	2.31	29.70	0.52	0.38	 32.20
std	        257.3	    0.49    	0.84	14.53	1.10	0.81	 49.69
min	        1.0	        0.0	        1.0	     0.42	0.0	    0.0	      0.00
25%	        223.5	    0.0	        2.0	    20.13	0.0	    0.0	      7.91
50%	        446.0	    0.0	        3.0	    28.00	0.0	    0.0	     14.45
75%	        668.5	    1.0	        3.0	    38.00	1.0	    0.0	     31.00
max	        891.0	    1.0	        3.0	    80.00	8.0	    6.0	    512.33
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
3) Age might be able to be inferred or imputed, althought 20% is a large number to impute
4) Embarked is categorical so can't be imputed -- ignore samples or ignore feature?
5) Ticket number is alphanumeric - unsure if there's anything in there. Ignore for now.
'''



'''
BASELINE

I'll start with a basic random forrest with:
1) Imputing the mean of Age in the missing samples
2) Not using Cabin feature
3) Ignoring 2 missing samples in Embarked
4) Encoding categorical variables
'''

# data prep
def prep_v0(data):
    """
    First go at preparing data for Random Forest
    :param data: DataFrame containing data to be prepared for the model
    :return: DataFrame of prepared data in the form read to go into the model
    """

    features = data.drop('Survived', axis=1).set_index('PassengerId')
    labels = data[['Survived', 'PassengerId']].set_index('PassengerId')

    # Sex to numeric
    features['Sex'] = features['is_female'].map(dict(male=0, female=1))
    # impute age
    imputed_age = features['Age'].mean()
    features['Age'].fillna(imputed_age)
    # ignore samples missing from Embarked
    features = features.loc[features['Embarked'].notnull(), :]
    # encode categorical columns
    categoricals = ['Embarked']
    a = pd.get_dummies(features, columns=categoricals)
    # remove unused
    drop_cols = ['Sex', 'Name', 'Ticket', 'Cabin'] + categoricals
    features.drop(drop_cols, axis=1, inplace=True)
    return features

train_v0 = prep_v0(train)
