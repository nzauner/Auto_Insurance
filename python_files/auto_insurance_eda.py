#!/usr/bin/env python
# coding: utf-8

# # Auto Insurance Analysis
# 
# ## Exploratory Data Analysis

# ### Library Import

# In[ ]:


#Import libraries
get_ipython().run_line_magic('run', '../python_files/imports')


# ## Data Import and Data Examination

# In[ ]:


# import auto insurance data
auto_df = pd.read_csv('../data/auto_insurance_data.csv')

# change column names to lower-case
auto_df.columns = [i.lower() for i in auto_df.columns]

# quick overview of the dataset
auto_df


# After a quick overview of the dataset, we see that we are working with 6043 total observations and 25 different variables. The response variable we will be using is Crash, which indicates whether a car was in a crash or not. The remaining 24 variables will be used as explanatory variables. We also notice a good mix of continuous and categorical variables.

# In[ ]:


# quick review of the variables in the dataset
auto_df.info()


# For modeling purposes, we know that we will have to convert all categorical variables to dummy variables. As we can see above, there are 10 categorical variables that will need to go through this conversion.

# In[ ]:


# quick review of the characteristics of our current continuous variables in the dataset
auto_df.describe()


# We notice above that there is a large range between some of our observations. However, it is not appropriate to dismiss these as outliers, as we do not want to skew or create bias within our dataset. Also, above we cannot view the descriptions of our 10 categorical variables until we convert them to continous variables.

# In[ ]:


# check the number of NaN values in the dataset
auto_df.isna().sum()


# Fortunately, we see above that our dataset does not contain any missing values, so we will not need to worry about imputation.

# ## Data Cleaning and Data Transformations

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


# In[ ]:


# Log Transformations for non-normalized variables. Drop the original variable from the dataset.

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
auto_df.describe()


# ## Initial Train and Test Dataset Creation

# In[ ]:


#Split auto_insurance_df into train and test datasets for our logistic and linear regression models

#'features' will be used in both models
features = auto_df.drop(['crash', 'crash_cost'], axis = 1)

#train and test datasets for logistic regression model
crash = auto_df['crash']
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(features, crash, test_size = 0.2, random_state = 10)

#train and test datasets for simple linear regression model
crash_cost = auto_df['crash_cost']
x_train_lin, x_test_lin, y_train_lin, y_test_lin = train_test_split(features, crash_cost, test_size = 0.2, random_state = 10)


# ## Data Exploration

# In[ ]:


# Correlations for logistic regression model
x_train_log.corr(method = 'pearson')


# In[ ]:


#Correlation Heatmap for logistic regression model

mask = np.zeros_like(x_train_log.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(35,30))
ax = sns.heatmap(x_train_log.corr(method='pearson'), cmap="coolwarm", mask=mask, annot=True, annot_kws={"size": 18}, square=True, linewidths=4)
sns.set_style('white')
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#plt.ylabel(ylabel=' ', labelpad=100)
plt.show()


# In[ ]:


# Correlations for simple linear regression model
x_train_lin.corr(method = 'pearson')


# In[ ]:


#Correlation Heatmap for simple linear regression model

mask = np.zeros_like(x_train_lin.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True

plt.figure(figsize=(35,30))
ax = sns.heatmap(x_train_lin.corr(method='pearson'), cmap="coolwarm", mask=mask, annot=True, annot_kws={"size": 18}, square=True, linewidths=4)
sns.set_style('white')
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14, rotation=0)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#plt.ylabel(ylabel=' ', labelpad=100)
plt.show()


# ## Feature Selection

# ### Recursive Feature Elimination for Logistic Regression Model

# In[ ]:


logreg_model = LogisticRegression()
rfecv_log = RFECV(estimator=logreg_model, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv_log.fit(x_train_log, y_train_log)


# In[ ]:


feature_importance_log = list(zip(features, rfecv_log.support_))
new_features_log = []
for key,value in enumerate(feature_importance_log):
    if(value[1]) == True:
        new_features_log.append(value[0])
        
print(new_features_log)


# ### Recursive Feature Elimination for Simple Linear Regression Model

# In[ ]:


linreg_model = LinearRegression()
rfecv_lin = RFECV(estimator=linreg_model, step=1, scoring='r2')
rfecv_lin.fit(x_train_lin, y_train_lin)


# In[ ]:


feature_importance_lin = list(zip(features, rfecv_lin.support_))
new_features_lin = []
for key,value in enumerate(feature_importance_lin):
    if(value[1]) == True:
        new_features_lin.append(value[0])
        
print(new_features_lin)


# ## Final Train and Test Datasets after Feature Selection

# In[ ]:


#final train and test datasets for logistic regression model
x_train_log = x_train_log[new_features_log]
x_test_log = x_test_log[new_features_log]

#final train and test datasets for simple linear regression model
x_train_lin = x_train_lin[new_features_lin]
x_test_lin = x_test_lin[new_features_lin]


# In[ ]:


print(x_train_log.shape)
print(x_test_log.shape)
print(x_train_lin.shape)
print(x_test_lin.shape)

