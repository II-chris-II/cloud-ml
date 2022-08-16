import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import datetime

# Loading the datasets
print('Loading datasets...')
df1 = pd.read_csv(
    'fraudTrain.csv', index_col=0)
df2 = pd.read_csv(
    'fraudTest.csv', index_col=0)

# Adding both datasets together with updated indexes
df = pd.concat([df1, df2], ignore_index=True)

# Sort the data by timestamp
df.sort_values('trans_date_trans_time', inplace=True)

print('Updating values...')
# Change the following to the correct data types
df['is_fraud'] = df['is_fraud'].astype('bool')
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
def dob2unix(dob):
    date_format = datetime.datetime.strptime(dob, "%Y-%m-%d")
    unix_time = datetime.datetime.timestamp(date_format)
    return unix_time
df['dob'] = df['dob'].transform(dob2unix)
def gender2binary(gender):
    return 1 if gender=='M' else 0
df['gender'] = df['gender'].transform(gender2binary)
df['category'] = df['category'].astype('category')

print('Preparing for ML stuff...')
# Split the dataset between test and train data
# feature_cols = ['unixtime', 'category',
#                 'amt', 'gender', 'zip', 'dob']
X = df[['amt', 'gender', 'zip', 'dob']].to_numpy()
y = df['is_fraud'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print('Doing ML stuff...')
# Create the model
# decision_tree = DecisionTreeClassifier(
#     criterion='entropy', max_depth=6, random_state=42, class_weight='balanced')
# model = decision_tree.fit(X_train, y_train)
log_reg = LogisticRegression(random_state=42, verbose=1)
model = log_reg.fit(X_train, y_train)

print('###### PREDICTION ######')
values = np.array([123,1,10001,682368231])
print(model.predict(values.reshape(1,-1)))

print('Saving model...')
# Saving model to disk
pickle.dump(model, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))

print('Done!')