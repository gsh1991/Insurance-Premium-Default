
# Importing the libraries
import pandas as pd # Data Manipulation
import numpy as np # NUmerical Calculation
from scipy.stats import chi2_contingency # Feature selection


# Loading the dataset
data = pd.read_excel("C:\\Users\\Gopinaath\\OneDrive\\Desktop\\IBM Project\\Insurance Dataset.xlsx")

# To find the top 5 rows
data.head()

# To find the last 5 rows
data.tail()

# To find the descriptive statistics
data.describe()

# To find the data types
data.dtypes

# To check for Balanced/Imbalanced Data

# count the number of records in target class
counts = data['Default'].value_counts()

# print the counts of each class
print(counts)

# calculate the percentage of records in each class
percentage = counts / len(data) * 100

# print the percentage of records in each class
print(percentage)


# Data preprocessing Steps

# Typecasting - Converting float into int types
data.PolicyholderAgeinyears = data.PolicyholderAgeinyears.astype('int64')
data.MonthlyPremium = data.MonthlyPremium.astype('int64')
data.PremiumPaidtilldate = data.PremiumPaidtilldate.astype('int64')
data.RiskScore = data.RiskScore.astype('int64')

# To find the data types
data.dtypes

## Identify duplicates records in the data
duplicate = data.duplicated()
duplicate
sum(duplicate)

#### Outlier Treatment ####
import seaborn as sns

# Let's find outliers in numerical columns
sns.boxplot(data.PolicyholderAgeinyears) # No outliers
sns.boxplot(data.IncomeofPolicyholderperannum) # Outlier present
sns.boxplot(data.Dependents) # No outliers
sns.boxplot(data.PolicyTerminyears) # Outlier present
sns.boxplot(data.Sumassured)# Outlier present
sns.boxplot(data.YearlyPremium)# Outlier present
sns.boxplot(data.MonthlyPremium)# Outlier present
sns.boxplot(data.NumberofPremiumspaidinyears) # No outliers
sns.boxplot(data.NumberofPremiumspaidtilldateinmonths)# No outliers
sns.boxplot(data.PremiumPaidtilldate)# No outliers
sns.boxplot(data.latepayment0to3months)# Outlier present
sns.boxplot(data.latepayment3to6months) #Outlier present
sns.boxplot(data.latepayment6to9months) #Outlier present
sns.boxplot(data.latepayment9to12months) #Outlier present
sns.boxplot(data.Morethan12monthsdelay) #Outlier present
sns.boxplot(data.Totaldelayedmonths)#Outlier present
sns.boxplot(data.RiskScore)#Outlier present

# Outlier Treatment using IQR & Winsorization

IQR = data['IncomeofPolicyholderperannum'].quantile(0.75) - data['IncomeofPolicyholderperannum'].quantile(0.25)

lower_limit = data['IncomeofPolicyholderperannum'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['IncomeofPolicyholderperannum'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
#pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['IncomeofPolicyholderperannum'])

data_t = winsor.fit_transform(data[['IncomeofPolicyholderperannum']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t.IncomeofPolicyholderperannum)
data["IncomeofPolicyholderperannum"] = data_t["IncomeofPolicyholderperannum"]
sns.boxplot(data.IncomeofPolicyholderperannum)

IQR = data['PolicyTerminyears'].quantile(0.75) - data['PolicyTerminyears'].quantile(0.25)
lower_limit = data['PolicyTerminyears'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['PolicyTerminyears'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['PolicyTerminyears'])

data_t1 = winsor.fit_transform(data[['PolicyTerminyears']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t1.PolicyTerminyears)
data["PolicyTerminyears"] = data_t1["PolicyTerminyears"]
sns.boxplot(data.PolicyTerminyears)

IQR = data['Sumassured'].quantile(0.75) - data['Sumassured'].quantile(0.25)
lower_limit = data['Sumassured'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Sumassured'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['Sumassured'])

data_t2 = winsor.fit_transform(data[['Sumassured']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t2.Sumassured)
data["Sumassured"] = data_t2["Sumassured"]
sns.boxplot(data.Sumassured)

IQR = data['YearlyPremium'].quantile(0.75) - data['YearlyPremium'].quantile(0.25)

lower_limit = data['YearlyPremium'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['YearlyPremium'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['YearlyPremium'])

data_t3 = winsor.fit_transform(data[['YearlyPremium']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t3.YearlyPremium)
data["YearlyPremium"] = data_t3["YearlyPremium"]
sns.boxplot(data.YearlyPremium)

IQR = data['MonthlyPremium'].quantile(0.75) - data['MonthlyPremium'].quantile(0.25)

lower_limit = data['MonthlyPremium'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['MonthlyPremium'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['MonthlyPremium'])

data_t4 = winsor.fit_transform(data[['MonthlyPremium']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t4.MonthlyPremium)
data["MonthlyPremium"] = data_t4["MonthlyPremium"]
sns.boxplot(data.MonthlyPremium)

IQR = data['latepayment0to3months'].quantile(0.75) - data['latepayment0to3months'].quantile(0.25)

lower_limit = data['latepayment0to3months'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['latepayment0to3months'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['latepayment0to3months'])

data_t5 = winsor.fit_transform(data[['latepayment0to3months']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t5.latepayment0to3months)
data["latepayment0to3months"] = data_t5["latepayment0to3months"]
sns.boxplot(data.latepayment0to3months)

IQR = data['latepayment3to6months'].quantile(0.75) - data['latepayment3to6months'].quantile(0.25)
lower_limit = data['latepayment3to6months'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['latepayment3to6months'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['latepayment3to6months'])

data_t6 = winsor.fit_transform(data[['latepayment3to6months']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t6.latepayment3to6months)
data["latepayment3to6months"] = data_t6["latepayment3to6months"]
sns.boxplot(data.latepayment3to6months)

IQR = data['latepayment6to9months'].quantile(0.75) - data['latepayment6to9months'].quantile(0.25)
lower_limit = data['latepayment6to9months'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['latepayment6to9months'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['latepayment6to9months'])

data_t7 = winsor.fit_transform(data[['latepayment6to9months']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t7.latepayment6to9months)
data["latepayment6to9months"] = data_t7["latepayment6to9months"]
sns.boxplot(data.latepayment6to9months)

IQR = data['latepayment9to12months'].quantile(0.75) - data['latepayment9to12months'].quantile(0.25)
lower_limit = data['latepayment9to12months'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['latepayment9to12months'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['latepayment9to12months'])

data_t8 = winsor.fit_transform(data[['latepayment9to12months']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t8.latepayment9to12months)
data["latepayment9to12months"] = data_t8["latepayment9to12months"]
sns.boxplot(data.latepayment9to12months)

IQR = data['Morethan12monthsdelay'].quantile(0.75) - data['Morethan12monthsdelay'].quantile(0.25)
lower_limit = data['Morethan12monthsdelay'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Morethan12monthsdelay'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['Morethan12monthsdelay'])

data_t9 = winsor.fit_transform(data[['Morethan12monthsdelay']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t9.Morethan12monthsdelay)
data["Morethan12monthsdelay"] = data_t9["Morethan12monthsdelay"]
sns.boxplot(data.Morethan12monthsdelay)

IQR = data['Totaldelayedmonths'].quantile(0.75) - data['Totaldelayedmonths'].quantile(0.25)
lower_limit = data['Totaldelayedmonths'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Totaldelayedmonths'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['Totaldelayedmonths'])

data_t10 = winsor.fit_transform(data[['Totaldelayedmonths']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t10.Totaldelayedmonths)
data["Totaldelayedmonths"] = data_t10["Totaldelayedmonths"]
sns.boxplot(data.Totaldelayedmonths)

IQR = data['RiskScore'].quantile(0.75) - data['RiskScore'].quantile(0.25)
lower_limit = data['RiskScore'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['RiskScore'].quantile(0.75) + (IQR * 1.5)
############### Winsorization ###############
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['RiskScore'])

data_t11 = winsor.fit_transform(data[['RiskScore']])

# Inspect the minimum caps and maximum caps
# winsor.left_tail_caps_, winsor.right_tail_caps_

# Let's see boxplot
sns.boxplot(data_t11.RiskScore)
data["RiskScore"] = data_t11["RiskScore"]
sns.boxplot(data.RiskScore)

## Missing Values imputation
# Check for count of NA's in each column
data.isna().sum()
# There are no missing values, hence imputation need not to be performed

#Dummy variable creation
data.columns
data.shape
data.dtypes

data.IncomeofPolicyholderperannum = data.IncomeofPolicyholderperannum.astype('int64')
data.PolicyTerminyears = data.PolicyTerminyears.astype('int64')
data.YearlyPremium = data.YearlyPremium.astype('int64')
data.MonthlyPremium = data.MonthlyPremium.astype('int64')
data.latepayment0to3months = data.latepayment0to3months.astype('int64')
data.latepayment3to6months = data.latepayment3to6months.astype('int64')
data.latepayment6to9months = data.latepayment6to9months.astype('int64')
data.latepayment9to12months = data.latepayment9to12months.astype('int64')
data.Morethan12monthsdelay = data.Morethan12monthsdelay.astype('int64')
data.Totaldelayedmonths = data.Totaldelayedmonths.astype('int64')
data.RiskScore = data.RiskScore.astype('int64')
data.Sumassured = data.Sumassured.astype('int64')
data.dtypes

# Creating Labelencoder for categorical columns
from sklearn.preprocessing import LabelEncoder
# Creating instance of labelencoder
labelencoder = LabelEncoder()
data["MaritalStatus"] = labelencoder.fit_transform(data["MaritalStatus"])
data["PolicyholderName"] = labelencoder.fit_transform(data["PolicyholderName"])
data["Nominee"] = labelencoder.fit_transform(data["Nominee"])
data["Accomodation"] = labelencoder.fit_transform(data["Accomodation"])
data["ResidenceAreaType"] = labelencoder.fit_transform(data["ResidenceAreaType"])
data["PlanName"] = labelencoder.fit_transform(data["PlanName"])
data.dtypes

# Feature selection by Calculating Information Value to tell the predictive power of all variables in relation to default variable for removing unwanted variables

# Define function to calculate IV
def calc_iv(df, feature, target):
    lst = []
    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 0)].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])
    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Non_Default', 'Default'])
    data['Default_Rate'] = data['Default'] / data['All']
    data['Non_Default_Rate'] = (data['Non_Default']) / (data['All'])
    data['WoE'] = np.log(data['Non_Default_Rate'] / data['Default_Rate'])
    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})
    data['IV'] = np.sum((data['Non_Default_Rate'] - data['Default_Rate']) * data['WoE'])
    return data['IV'].values[0]

# Compute IV for all variables
target_var = 'Default' # name of the target variable
iv_values = {}
for feature in data.columns:
    if feature != target_var:
        iv_values[feature] = calc_iv(data, feature, target_var)

# Print IV values in descending order
for feature, iv in sorted(iv_values.items(), key=lambda x: x[1], reverse=True):
    print(f'{feature}: {iv:.4f}')
    
# Drop features with less informative value gained from Chi-Squared test
data.drop(['NumberofPremiumspaidtilldateinmonths', 'PolicyTerminyears','Sumassured','YearlyPremium','MonthlyPremium','latepayment0to3months','Accomodation','Nominee','Dependents','MaritalStatus','ResidenceAreaType','latepayment3to6months','latepayment6to9months','latepayment9to12months','Morethan12monthsdelay','RiskScore','PolicyholderID','PolicyholderName','PlanName'], axis = 1, inplace = True)
data.dtypes
data.columns
data.shape


# Model Building using Decision Tree algorithm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# Split data into training and testing sets
X = data.drop(columns=["Default"])
y = data["Default"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the performance of the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))





