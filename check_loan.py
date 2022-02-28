from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, ExtraTreeClassifier

iris = load_iris()

df = pd.read_csv(path2)

df.head()

for col in df.columns:
  if col == "bad_credit":continue
  if df[col].dtype != object:
    skew_value = df[col].skew()
    if skew_value>0.7:
      print(col, skew_value)
      df[col] = np.log1p(df[col])
df.head()

from sklearn.preprocessing import LabelEncoder
for col in df.columns:
    le = LabelEncoder()
    if df[col].dtype == object:
      df[col] = le.fit_transform(df[col])
      df[col] = df[col].astype('str')

Y = df['bad_credit']
X = df.drop(['bad_credit'], axis=1)
try:
  df = df.drop(['customer_id'], axis=1)
except:
  pass
X.info()

print(X.shape)
try:
  X = X.drop(["foreign_worker"], axis=1)
except:
  pass

print(X.shape)
X.head()

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(X, Y)

dt = DecisionTreeClassifier(criterion='entropy', random_state=111, min_samples_leaf=2, 
                            class_weight={1:0.6, 0:0.4},max_depth=3, min_samples_split=2)
dt.fit(trainX,trainY)

test_pred = dt.predict(testX)
train_pred = dt.predict(trainX)

from sklearn.metrics import classification_report
print(classification_report(testY,test_pred))
print(classification_report(trainY,train_pred))

fi_dict = {}
fi = dt.feature_importances_
cols = X.columns
for i in range(len(cols)):
  fi_dict[cols[i]] = fi[i]

fi_dict = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
fi_dict

export_graphviz(dt,'dt.tree')












      
      
      

