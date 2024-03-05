from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

try: 
    df = pd.read_csv('IGN.csv', 
                     names=['title', 'score','score_phrase','platform','genre','release_year','release_month','release_day'])
except Exception as e:
    print("Error: ", e)

df_new = df.dropna().drop_duplicates()
df_new['score'] = pd.to_numeric(df['score'], errors='coerce') 
df_new['release_month'] = pd.to_numeric(df['release_month'], errors='coerce')

# linear regression to help understand the relationship between video game scores and platform, genre and release year.

df_new = df_new.dropna(subset=['score']) # Needed otherwise error, despite the previous dropna
X = df_new[['platform', 'genre', 'release_year']]  
y = df_new['score']

categorical_features = ['platform', 'genre']
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', one_hot_encoder, categorical_features)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

#----------------------------------------------------------------------------------------------------------------
# Association Rule Mining to investigate if certain combinations of genre and platform are associated with higher scores

df_new['score'] = pd.cut(df_new['score'], bins=[0, 6, 8, 10], labels=["Low", "Medium", "High"])
df_new['genre'] = df_new['genre'].apply(lambda x: x.split(', ')) 
df_new['platform'] = df_new['platform'].apply(lambda x: [x])  

df_new['items'] = df_new[['platform', 'genre', 'score']].apply(lambda x: x['platform'] + x['genre'] + [x['score']], axis=1)
df_new['items'] = df_new['items'].apply(lambda x: [str(i) for i in x])

transactions = df_new['items'].tolist()

encoder = TransactionEncoder()
onehot = encoder.fit_transform(transactions)
onehot_df = pd.DataFrame(onehot, columns=encoder.columns_)

frequent_itemsets = apriori(onehot_df, min_support=0.01, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False).to_string())
