import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
from re import match

########################################################################################################################

'''
Step 0: Import data and check which values are missing
'''

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

########################################################################################################################

'''
STEP 1: Basic model

I'll start with a basic random forrest with:
1) Imputing the mean of training set features where there are missing values (Age)
2) Not using Cabin feature
3) Ignoring 2 missing samples in Embarked
4) One-hot encoding categorical variables
'''

# data prep
def prep_step1(data, test_or_train='train', imputer=None):
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
    categorical_columns = ['Embarked', 'Pclass']
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
X_train, y_train, imputer = prep_step1(train)
X_test, _, _ = prep_step1(test, test_or_train='test', imputer=imputer)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_test = pd.DataFrame(zip(X_test.index, model.predict(X_test)), columns=['PassengerId', 'Survived'])
y_test.to_csv('data/predictions_1.csv', index=False)
'''
Accuracy for predictions_1.csv:  0.75837
'''

########################################################################################################################

'''
Step 2: Can any of the variables be more descriptive?
Which are most important to current model?
'''

feature_importance_rf = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importance_rf.round(3).to_clipboard(sep='\t')
'''
Feature     Importance
Age	            0.264
is_female	    0.259
Fare	        0.255
Pclass_3	    0.057
SibSp	        0.046
Parch	        0.038
Pclass_1	    0.029
Pclass_2	    0.017
Embarked_S	    0.015
Embarked_C	    0.012
Embarked_Q	    0.008

Age, Fare, and is_female are important to model.
Focus on improving data quality of those first, and explore ignoring the others.
'''

#apply SelectKBest class to extract 5 best features
best_features = SelectKBest(score_func=chi2, k=5).fit(X_train, y_train)
best_features = pd.DataFrame(zip(best_features.scores_, best_features.pvalues_),
                             index=X_train.columns,
                             columns=['score', 'pvalue']).sort_values(by='score', ascending=False)
best_features.round(3).to_clipboard(sep='\t')
'''
Feature        score   pvalue
Fare	    4453.395	0.000
is_female	 169.242	0.000
Pclass_1	  53.819	0.000
Pclass_3	  40.799	0.000
Age	          27.973	0.000
Embarked_C	  20.829	0.000
Parch	      10.449	0.001
Pclass_2	   6.363	0.012
Embarked_S	   5.644	0.018
SibSp	       2.391	0.122
Embarked_Q	   0.017	0.897

Again, Fare, is_female, & Age are important here. This highlights being in Pclass 1 or 3 as important. 
All but 2 scores are significant.
'''

# check feature correlation with survival
correlation_with_survival = X_train.corrwith(y_train).sort_values()
correlation_with_survival.round(3).to_clipboard(sep='\t')
'''
Pclass_3	-0.320
Embarked_S	-0.152
Age	        -0.075
SibSp	    -0.034
Embarked_Q	 0.005
Parch	     0.083
Pclass_2	 0.095
Embarked_C	 0.17
Fare	     0.255
Pclass_1	 0.282
is_female	 0.542

None show particularly high correlation with Survived. Although the most correlated features tend to agree with
what is observed above. 
'''

'''
Step 2 conclusions: Fare, is_female, Pclass_1, Pclass_3, & Age are most important features.
Focus on improving data quality of those first.
For the others, explore ignoring the others, or improving data quality too.

Age: lots of missing data - can we infer something about the age from other features?
Fare: can we infer the missing fare in the output from other features?
'''

########################################################################################################################

'''
Step 3: Improving relevant features -- Fare
'''

# check skew of variables
# ax = sns.pairplot(train)
# ax.savefig('plots/pairplot.png')  # pair plot to see visualise how features are distributed and related to eachother
skew_fare = train['Fare'].skew().round(2)
'''
Fare: strong leftward skew (4.79) -> use Log(Fare) instead?
'''

'''
Log(Fare) won't work because there are 0 values. Try Log(Fare + 1), as distribution is still preserved.
'''
X_train['Log_Fare'] = (X_train['Fare'] + 1).apply(np.log)  # determine log of Fare + 1
skew_log_fare = X_train['Log_Fare'].skew().round(3)  # get skew
# plot grid with distribution plots of Fare and Log_Fare for comparison
# grid = sns.FacetGrid(pd.melt(X_train[['Fare', 'Log_Fare']], var_name='domain', value_name='value'),
#                      col='domain', sharex=False, sharey=False, size=4)
# grid.map(sns.distplot, 'value', bins=50, hist=True, kde=True).set_titles("{col_name}").set_axis_labels(x_var='Density')
# grid.savefig('plots/fare-vs-log_fare.png')
'''
Log_Fare: skew reduced significantly (0.4).
Apply to test set and see if accuracy is improved:
'''
# get fresh set of prepared data
X_train, y_train, imputer = prep_step1(train)
X_test, _, _ = prep_step1(test, test_or_train='test', imputer=imputer)
# transform into log domain
X_train['Log_Fare'] = (X_train['Fare'] + 1).apply(np.log)
X_test['Log_Fare'] = (X_test['Fare'] + 1).apply(np.log)
# remove old columns
X_train.drop('Fare', axis=1, inplace=True)
X_test.drop('Fare', axis=1, inplace=True)
# fit model and predict
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_test = pd.DataFrame(zip(X_test.index, model.predict(X_test)), columns=['PassengerId', 'Survived'])
y_test.to_csv('data/predictions_2.csv', index=False)
'''
Accuracy: 0.74401
Worse!

Step 3 Conlcusion: putting Fare into Log Domain gives no improvement of accuracy
'''

########################################################################################################################

'''
Step 4: Improving relevant features -- Age
Age: correlated with anything? Title maybe? (e.g. Mr, Mrs, Master, Miss). Perhaps can do better imputing with that.
'''
# get fresh set of data
train = pd.read_csv('data/train.csv').set_index('PassengerId')
test = pd.read_csv('data/test.csv').set_index('PassengerId')
X_train, y_train, imputer = prep_step1(train)
X_test, _, _ = prep_step1(test, test_or_train='test', imputer=imputer)

# extract title from name
def title_extractor(name):
    # extract title from name
    mtch = match(r'.+, ([a-zA-Z ]+)\. .+', name)
    title = mtch.groups()[0]
    return title
train['Title'] = train['Name'].apply(title_extractor)
# update prepared data with title and one-hot-encode
X_train['Title'] = train.loc[X_train.index, 'Title']
X_train = pd.get_dummies(X_train, columns=['Title'])
'''
What correlates with Age?
'''
all_but_age = [col for col in X_train.columns if col != 'Age']  # extract all columns except Age
correlation_with_age = X_train[all_but_age].corrwith(X_train['Age']).sort_values()  # correlate Age with other columns
correlation_with_age.round(3).to_clipboard(sep='\t')
'''
Feature         Corr. Coef.
Title_Master	   -0.378
Pclass_3           -0.279
Title_Miss	       -0.249
SibSp	           -0.232
Parch	           -0.178
is_female	       -0.089
Embarked_S	       -0.021
Title_Mlle	       -0.021
Title_Mme	       -0.015
Embarked_Q	       -0.013
Title_Ms	       -0.004
Title_the Countess	0.009
Pclass_2	        0.009
Title_Jonkheer	    0.022
Title_Don	        0.027
Embarked_C	        0.034
Title_Lady	        0.048
Title_Sir	        0.05
Title_Major	        0.069
Title_Dr	        0.073
Title_Rev	        0.086
Fare	            0.089
Title_Col	        0.104
Title_Capt	        0.104
Title_Mrs	        0.161
Title_Mr	        0.191
Pclass_1	        0.316

Nothing especially, Maybe only have a title of Master helps.
Will ignore all Titles except Title_Master and try a different imputing strategy.
'''
# determine Title_XXX columns to drop
drop_cols = [col for col in X_train.columns if (col != 'Title_Master') and (col[:5] == 'Title')]
# drop them
X_train.drop(drop_cols, axis=1, inplace=True)
