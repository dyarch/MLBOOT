
# coding: utf-8

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#settings
pd.options.display.max_columns = 250
pd.options.display.max_rows = 250
sns.set(style="whitegrid")

import warnings
warnings.simplefilter('ignore')


# # Importing Data
# 
# > Importing data into CSV file from the database may sometime lead to loss of data type and corruption of data. Most common reason would be presence of 'comma' character in the data. Double check your data for any irregularities.

# In[2]:


df = pd.read_pickle("C:\Git Repos\OPIMLBoot\wowpickle.pkl")
# Picking 73,000 records randomly as the data set is very big. Dont use the following line while doing proper analysis.
df = df.sample(73000)


# In[3]:


print("Rows: ", df.shape[0])
print("Columns: ", df.shape[1])


# ### Dropping duplicates (if any)

# In[4]:


print("Before dropping duplicates: ")
print("")
print(df.info())
print("")
print("After dropping duplicates(if any): ")
print("")
df6 = df.drop_duplicates()
print(df6.info())
df = df6.copy()


# ## Description of the data - Statistics

# ### Converting columns that have textual data to upper case.

# In[5]:


# Converting column values to upper case
for item in df.columns:
    if df[item].dtype == 'object':
        df[item] = df[item].str.upper()


# In[6]:


df.sample(5)


# ### Target value analysis
# ##### Replace 'UHGPrimacy' with column name of your target feature, to get data break down for each class of your target feature.

# In[7]:


print(df.UHGPrimacy.value_counts())
ax = sns.countplot(df.UHGPrimacy,label="Count")


# ##### If the problem is a regression problem i.e. we have to predict a continous variable like house price, then the following graph would be more useful

# In[8]:


print(df.UHGPrimacy.describe())
sns.distplot(df.UHGPrimacy)


# ## Now, we will look at other features in the dataset and get a high level understanding of what kind of data they hold.

# #### Below is a table giving a brief description about the variance of data in each column.

# In[9]:


new_df = pd.DataFrame(columns=['ColumnName', 'Null_Count_%', 'Distinct_Values', 'Min', 'Mean', 'Max', 'dtype'])
i = 0
#print(df.shape[0])
for item in df.columns:
    if(df[item].dtype == 'object'):
        new_df.loc[i] = [item, df[item].isnull().sum()/df.shape[0] * 100, df[item].nunique(), "Na", "Na", "Na", df[item].dtype]
    else:
        new_df.loc[i] = [item, df[item].isnull().sum()/df.shape[0] * 100, df[item].nunique(), df[item].min(), df[item].mean(), df[item].max(), df[item].dtype]
    
    i=i+1
print("Column Information: ")
new_df


# #### List of column that have more than 60% null values. General recommendation is to drop columns that contain more than 60% null value

# In[10]:


# removing column with large number of missing values
count = 0
dropped_index = []
#print(df.isnull().sum())
dropped_columns = pd.DataFrame(columns=['ColumnName', 'Null_Count_%', 'Distinct_Values', 'Min', 'Mean', 'Max', 'dtype'])
for index, row in new_df.iterrows():
    null_count = row['Null_Count_%']
    if null_count > 60.0:
        count = count + 1
        df = df.drop(columns=[row['ColumnName']])
        dropped_columns = dropped_columns.append(row, ignore_index=True)
        dropped_index.append(index)
        #new_df.drop(new_df.index[[index]], inplace=True)
        
new_df.drop(dropped_index, inplace=True)
print("Number of Features Eliminated: ", count)
print("Eliminated Columns: ")
dropped_columns


# #### Encoding and Filling missing values
# > We have divided the data into three categories: <br>
# - Numerical
# - Textual/Categorical
# - Date

# ##### Sorting columns into their respective category

# In[11]:


# sorting into categorical and numerical list using dtype i.e. data type of the column
categorical_list = []
numerical_list = []
for i in df.columns.tolist():
    if df[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)

print('Number of categorical features:', str(len(categorical_list)))

new_df[new_df.ColumnName.isin(categorical_list)]


# #### Getting date columns from categorical list. For this we identify patterns that date column have and utilise that to generate our date_list.

# In[12]:


date_list = []
new_list2 = []
print("No. of categorical columns before removing date columns: ", len(categorical_list))
for item in categorical_list:
    if item[-4:] == 'DATE' or item[-3:] == '_DT' or item[-3:] == 'DOB':
        date_list.append(item)
    else:
        new_list2.append(item)
    
print("No. of categorical columns after removing date columns: ", len(new_list2))
categorical_list = new_list2

new_df[new_df.ColumnName.isin(date_list)]


# In[13]:


df[date_list].head()


# ## DATE ENCODING: START

# In[14]:


# removing run_date because it's format is different from that of date
date_list.remove('RUN_DATE')
date_list


# ##### Below is a function that convert dates to their numeric equivalent - for example - 1987-12-20 becomes 19871220. This enables machine learning models to compare date columns and extract useful information

# In[15]:


def encode_date_column(x):
    if x == None:
        date = '00000000'
    else:
        year = x[0:4]
        month = x[5:7]
        day = x[8:10]
        date = year+month+day
    return(date)


# In[16]:


# applying encoding function to columns present in date list
for item in date_list:
    df[item] = df[item].apply(encode_date_column)


# ## DATE ENCODING - END

# In[17]:


df[date_list].head()


# ## Encoding categorical values using label encoder

# In[18]:


# using label encoder for encoding categorical values
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

#dictionary for storing label encoder for each feature
le_dict = defaultdict(LabelEncoder)

for item in categorical_list:
    df[item] = le_dict[item].fit_transform(df[item].astype(str))

df[categorical_list].sample(5)


# In[19]:


# numerical column information
new_df[new_df.ColumnName.isin(numerical_list)]


# ### Filling null values in a column with the median value for the whole column

# In[20]:


from sklearn.preprocessing import Imputer

df1 = pd.DataFrame(Imputer(strategy='median').fit_transform(df))
df1.columns = df.columns
df1.index = df.index
df = df1.copy()


# ## Removing features having high variance
# #### Here we identified features that high number of distinct values i.e. feature that have unique value for each record. We remove such features they do not provide the model with any useful information. Example of such features - Name, SSN, Phone number etc. You should identify such features and remove them from your dataset. These can be identified by observing the 'Distinct_Value' column of the 'new_df' data frame.
# > new_df

# In[21]:


del df['MBRKEY']
del df['MEMBERID']
del df['FIRSTNAME']
del df['LASTNAME']
del df['SSN']
del df['FAMILYID']
del df['CARDCODEALT']
del df['ADDRESSLINE1']
del df['HOMEPHONENUM']
del df['SUBID']
del df['SUBSSN']
del df['SUBADDRESSLINE1']
del df['COMMENTSINT']
del df['INVENTORYROWID']
del df['SEQ_NBR']


# In[22]:


# Dividing the data set into features and target. Here we have 'UHGPrimacy' as our targer value.'y' - target value,
# i.e. the value which we have to predict.
y = df.UHGPrimacy
X = df.drop('UHGPrimacy', axis = 1)
feature_name = X.columns.tolist()


# # Feature Voting
# ### One of the most important aspect of machine learing is feature engineering. Feature engineering requires data analysis and business knowledge to identify which feature are important. Below we have utilised statistical and machine learning models to extract important feature and rated them.

# ## Pearson Correlation

# In[23]:


def cor_selector(X, y):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-20:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


# In[24]:


cor_support, cor_feature = cor_selector(X, y)
print(str(len(cor_feature)), 'selected features')


# ## Chi-2

# In[25]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=20)
chi_selector.fit(X_norm, y)


# In[26]:


chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')


# ## Recursive Feature Elimination

# In[27]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=20, step=5, verbose=5)
rfe_selector.fit(X_norm, y)


# In[28]:


rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')


# ## SelectFromModel

# ### Logistic Regression

# In[29]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), '1.3*median')
embeded_lr_selector.fit(X_norm, y)


# In[30]:


embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')


# ### Random Forest

# In[31]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold='1.1*median')
embeded_rf_selector.fit(X, y)


# In[32]:


embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


# ## LightGBM

# In[33]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

embeded_lgb_selector = SelectFromModel(lgbc, threshold='1.2*median')
embeded_lgb_selector.fit(X, y)


# In[34]:


embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')


# ## Conclusion

# In[35]:


pd.set_option('display.max_rows', None)
# put all selection together
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 
                                    'Logistic Regression': embeded_lr_support, 'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(20)


# ### Select features on the basis of the voting
# ##### For example : we pick features that have more than 3 votes

# In[36]:


selected_feature = feature_selection_df.loc[feature_selection_df['Total'] > 3]['Feature']


# #### Construct new data set containing selected features and store it as a pickle file for further analysis or training your model.

# In[37]:


df_selected = df[selected_feature].copy()
df_selected.head()


# In[38]:


df_selected.to_pickle("selected_feature_data.pkl")


# #### Store the selected feature list in a file

# In[39]:


with open('selected_feature_list.txt', 'w') as f:
    for s in selected_feature:
        f.write(str(s) + "\n")


# ## A function that proiveds alternate method for encoding dates.

# ##### For encoding date type we will choose a reference date column, in this case - 'LOAD_DT'. You could choose your own reference date column. In our case we choose this column as it represents when a claim loads into our system. Now we will find no. of days between 'load_dt' and our targer date of our column ( this provides useful information as - how many days till the expiry of a insurance provides better info to ML model than expiry date). There are many ways to handle date columns and the performance of these methods vary situation to situation.
# > <strong>call the function in the following way - [Replace the DATE ENCODING block with this code]</strong> <br>
# > > df = encode_date_columns_version_other(df, date_list)

# In[15]:


def encode_date_columns_version_other(df, date_list):
    # converting data type of date columns to datetime format.
    # errors='coerce' converts date that are above the max_date in python to NaT i.e. not a time
    
    # column 'RUN_DATE' has different format from other columns so we will handle it differently and remove it from date_list
    df.RUN_DATE = pd.to_datetime(df.RUN_DATE, errors='coerce', format='%Y%m%d')
    date_list.remove('RUN_DATE')
                   
    # encoding date_list columns to datatime data type
    for item in date_list:
        df[item] = pd.to_datetime(df[item], errors='coerce')
    
    # appending RUN_DATE to date_list for further processing
    date_list.append('RUN_DATE')
    
    # we will replace NaT with the max time
    for item in date_list:
        df[item] = df[item].replace(pd.NaT, pd.Timestamp.max)

    date_reference_column = 'LOAD_DT'
    # removing reference date column from date_list and we will later delete it from our data frame.
    date_list.remove('LOAD_DT')

    # replacing dates with no.of days(date in questin - load date)
    for item in date_list:
        df[item] = (df[item] - df[date_reference_column]).dt.days

    del df[date_reference_column]
    
    return df

