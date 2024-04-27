import pandas as pd 
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# Import dataset 
df = pd.read_csv('Data/delay.csv')

# Label Encoding
le_carrier = LabelEncoder()
df['carrier'] = le_carrier.fit_transform(df['carrier'])

le_dest = LabelEncoder()
df['dest'] = le_dest.fit_transform(df['dest'])

le_origin = LabelEncoder()
df['origin'] = le_origin.fit_transform(df['origin'])

# Converting Pandas DataFrame into a Numpy array
X = df.iloc[:, 0:6].values 
y = df['delayed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=18) 

# Create XGBoost Classifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Saving model to disk
y_pred = xgb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
pickle.dump(xgb, open('xgb_model.pkl', 'wb'))

# Load XGBoost model
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
