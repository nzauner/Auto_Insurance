#!/usr/bin/env python
# coding: utf-8

# # Auto Insurance Analysis
# 
# ## Exploratory Data Analysis

# ## Project Goals
# 
# - Analyze auto insurance data.
# - Build a logistic regression model to predict crash probability for auto insurance customers.
# - Build a linear regression model to predict crash cost for auto insurance customers.
# - Use model results to develop crash percentage, assign customers to new risk profiles, and risk probability percentages.
# - Determine cost of premiums based on customer risk profiles and risk probability percentages.

# ## Summary of Data
# 
# The dataset for this project contains 6044 records of auto insurance data. Each record
# represents a customer at an auto insurance company. Using this data, we will be able to ascertain what
# influences the likelihood of a car crash. Then subsequently, we will be able to determine the cost to resolve a claim. The data in this project is the typical type of corporate data you would receive from a company in the insurance field-- a typical flat file from client records.

# ### Library Import

# In[ ]:


#Import libraries
# get_ipython().run_line_magic('run', '../python_files/imports')
from imports import *

# ## Data Import and Data Examination

# In[ ]:


# import auto insurance data
auto_df = pd.read_csv('../data/auto_insurance_data.csv')

# change column names to lower-case
auto_df.columns = [i.lower() for i in auto_df.columns]

# quick overview of the dataset
# auto_df


# After a quick overview of the dataset, we see that we are working with 6044 total observations and 25 different variables. The response variable we will be using is 'crash', which indicates whether a car was in a crash or not. The remaining 24 variables will be used as explanatory variables. We also notice a good mix of continuous and categorical variables.

# In[ ]:


# quick review of the variables in the dataset
# auto_df.info()


# For modeling purposes, we know that we will have to convert all categorical variables to dummy variables. As we can see above, there are 10 categorical variables that will need to go through this conversion.

# In[ ]:


# quick review of the characteristics of our current continuous variables in the dataset
# auto_df.describe()


# We notice above that there is a large range between some of our observations. However, it is not appropriate to dismiss these as outliers, as we do not want to skew or create bias within our dataset. Also, above we cannot view the descriptions of our 10 categorical variables until we convert them to continous variables.

# In[ ]:


# check the number of NaN values in the dataset
# auto_df.isna().sum()


# Fortunately, we see above that our dataset does not contain any missing values, so we will not need to worry about imputation.

# ## Data Cleaning, Data Transformations, and Data Exploration
# 
# Below, we created dummy variables for our 10 categorical variables: mstatus, sex, parent1, red_car, revoked, urbanicity, education, job, car_use, and car_type. Using the mapping technique, these changes were appended to the dataset, and therefore, we did not have to drop any variables.

# In[ ]:


# Create dummy values for the categorical variables

auto_df['mstatus'] = auto_df['mstatus'].map({'Yes': 1, 'No': 0})
auto_df['sex'] = auto_df['sex'].map({'M': 1, 'F': 0})
auto_df['parent1'] = auto_df['parent1'].map({'Yes': 1, 'No': 0})
auto_df['red_car'] = auto_df['red_car'].map({'yes': 1, 'no': 0})
auto_df['revoked'] = auto_df['revoked'].map({'Yes': 1, 'No': 0})
auto_df['urbanicity'] = auto_df['urbanicity'].map({'Highly Urban/ Urban': 1, 'Highly Rural': 0})
auto_df['education'] = auto_df['education'].map({'<High School': 0, 'High School': 0, 'Bachelors': 1, 'Masters': 1, 'PhD': 1})
auto_df['job'] = auto_df['job'].map({'Student': 1, 'Blue Collar': 0, 'Clerical': 0, 'Doctor': 0, 'Home Maker': 0, 'Lawyer': 0, 'Manager': 0, 'Professional': 0})
auto_df['car_use'] = auto_df['car_use'].map({'Commercial': 1, 'Private': 0})
auto_df['car_type'] = auto_df['car_type'].map({'Sports Car': 1, 'SUV': 1, 'Minivan': 1, 'Pickup': 0, 'Van': 0, 'Panel Truck': 0})


# Next, we created log-transformed variables for our continuous variables that did not have normal distributions. Then, we dropped the original variables (the pre-transformed variables) from out dataset. This was performed on 3 of our feature variables: tif, bluebook, and travtime.

# In[ ]:


# Log Transformations for non-normalized variables. Then, drop the original variable from the dataset.

def log_col(df, col):
    '''Convert column to log values and
    drop the original column
    '''
    df[f'{col}_log'] = np.log(df[col])
    df.drop(col, axis=1, inplace=True)

log_col(auto_df, 'tif')
log_col(auto_df, 'bluebook')
log_col(auto_df, 'travtime')


# In[ ]:


# quick review of the characteristics of all variables in the dataset, 
# including the new dummy variables and log-transformed variables
# auto_df.describe()


# Our auto_df dataset is now ready for further evaluation. Above, we observe the newly edited variables from our dummy transformations and log-transformations. This leaves us with the same number of total observations and variable columns: 6044 observations and 25 variables (crash and crash_cost are our 2 response variables, and the remaining 23 variables are our feature variables).
# <p>
# Below, we explore the correlations between our response variables and feature variables. The correlation heatmap does a great job in providing a visual understanding of these relationships.

# In[ ]:


# Correlations between all variables in auto_df dataset
# auto_df.corr(method = 'pearson')


# In[ ]:


#Correlation Heatmap of all variables in auto_df dataset

# mask = np.zeros_like(auto_df.corr())
# triangle_indices = np.triu_indices_from(mask)
# mask[triangle_indices] = True

# plt.figure(figsize=(35,30))
# ax = sns.heatmap(auto_df.corr(method='pearson'), cmap="coolwarm", mask=mask, annot=True, annot_kws={"size": 18}, square=True, linewidths=4)
# sns.set_style('white')
# plt.xticks(fontsize=14, rotation=45)
# plt.yticks(fontsize=14, rotation=0)
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.show()


# ## Initial Train and Test Dataset Creation
# 
# In this section, we split the auto_df dataset into training and test datasets for modeling purposes, both for our logistic regression model and our simple linear regression model. We used an 80%-20% training and test split, and randomized the selection of the data pulled from the original dataset.
# 
# For our linear regression model, we did have to alter which data we were using because of our response variable crash_cost. Since crash_cost only gives us crash amounts for customers who DID get in accidents this past year, we must filter out the customers who did NOT get into car accidents this year. We can use this model to predict crash amounts for those who did get in a crash and might get into another accident. For the customers who have not been involved in an accident, we are not suggesting that it is impossible for them to get into a future accident. For those customers specifically, if they do get into a future accident, we can plug in their associated crash cost into our already established crash_cost model and then use that data to predict their future accident costs based on our feature variables. Since we do not have an associated crash_cost for these specific customers yet though, we cannot use them inside of this specific model. We can only add them to our model once they do in fact get into a crash (hopefully they don't!).

# In[ ]:


#Split auto_insurance_df into train and test datasets for our logistic and linear regression models

#train and test datasets for logistic regression model
crash = auto_df['crash']
features_log = auto_df.drop(['crash', 'crash_cost'], axis = 1)
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(features_log, crash, test_size = 0.2, random_state = 10)

#for our simple linear regression model, filter out customers who did NOT get into an accident this year, and create
# new dataframe consisting of only customers who DID get into an accident this year
customers_no_crash = auto_df[auto_df['crash'] == 0 ].index
auto_df.drop(customers_no_crash, inplace=True)

#train and test datasets for simple linear regression model
crash_cost = auto_df['crash_cost']
features_lin = auto_df.drop(['crash', 'crash_cost'], axis = 1)
x_train_lin, x_test_lin, y_train_lin, y_test_lin = train_test_split(features_lin, crash_cost, test_size = 0.2, random_state = 10)


# ## Feature Selection
# 
# For modeling purposes, we used recursive feature elimination for both our logistic regression model and our simple linear regression model. This process uses cross-validation techniques, using accuracy as a metric, to eliminate variables that may hurt our model performance. Those variables get dropped from the dataset prior to modeling.

# ### Recursive Feature Elimination for Logistic Regression Model

# In[ ]:


logreg_model = LogisticRegression()
rfecv_log = RFECV(estimator=logreg_model, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv_log.fit(x_train_log, y_train_log)


# In[ ]:


feature_importance_log = list(zip(features_log, rfecv_log.support_))
new_features_log = []
for key,value in enumerate(feature_importance_log):
    if(value[1]) == True:
        new_features_log.append(value[0])
        
# print(new_features_log)


# ### Recursive Feature Elimination for Simple Linear Regression Model

# In[ ]:


linreg_model = LinearRegression()
rfecv_lin = RFECV(estimator=linreg_model, step=1, min_features_to_select = 1, scoring='r2')
rfecv_lin.fit(x_train_lin, y_train_lin)


# In[ ]:


feature_importance_lin = list(zip(features_lin, rfecv_lin.support_))
new_features_lin = []
for key,value in enumerate(feature_importance_lin):
    if(value[1]) == True:
        new_features_lin.append(value[0])
        
# print(new_features_lin)


# ## Final Train and Test Datasets after Feature Selection
# 
# Here, we create our final training and test datasets that will be used for our modeling process. After reviewing the structure of each dataset for both of our models, we notice that our recursive feature elimination process removed 7 features for our logistic regression model data, giving us 16 features for this model. However, this process did not remove any features for our simple linear regression model data, leaving us with all 23 features for this model. Now, with our cleaned final datasets, we are ready to move onto the modeling phase of our study.

# In[ ]:


#final train and test datasets for logistic regression model
x_train_log = x_train_log[new_features_log]
x_test_log = x_test_log[new_features_log]

#final train and test datasets for simple linear regression model
x_train_lin = x_train_lin[new_features_lin]
x_test_lin = x_test_lin[new_features_lin]


# In[ ]:


# print(x_train_log.shape)
# print(x_test_log.shape)
# print(x_train_lin.shape)
# print(x_test_lin.shape)

