### Support Vector Classifier (SVC)

#یکی از مدل هایی که به خوبی عمل کرد مدل ماشین بردار پشتیبان می باشد
import joblib
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import clone



df = pd.read_csv('gtzan_5s_features.csv')
df_train = pd.read_csv("Data/gtzan_train.csv")
df_test = pd.read_csv("Data/gtzan_test.csv")

df.columns
df_train.columns

groups = df_train['file'].reset_index(drop=True)
X_train = df_train.drop(columns=["file", "genre", "segment", "start_sec"]).reset_index(drop=True)
y_train = df_train["genre"].reset_index(drop=True)


X_test = df_test.drop(columns=["file", "genre", "segment", "start_sec"])
y_test = df_test['genre']

     
cv = GroupKFold(n_splits=5, shuffle = True, random_state = 42)

# sanity check:
# for tr_idx, te_idx in cv.split(X_train, y_train, groups=groups):
#     print(f'training set index: {tr_idx}')
#     print('-'*100)
#     print(f'test set index{te_idx}')

model = Pipeline(
    steps=[
        ('standardScaler', StandardScaler()),
        ('model', SVC())
    ]
)

scores = cross_val_score(model, X_train, y_train, groups=groups,
                         cv=cv, scoring="f1_macro")
print(f"SVM CV: {scores.mean():.4f} (+/- {scores.std():.4f}), {scores.round(4)}")



model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm_svc = confusion_matrix(y_pred=y_pred, y_true=y_test)
sns.heatmap(cm_svc, annot=True, fmt='d')

f1_score(y_pred=y_pred, y_true=y_test, average='macro')


joblib.dump(model, 'saved_models/svc_model_raw.joblib')



## hyper parameter tuning for Support Vector Machine (classification)


hp_parameters = [
    {
        "model__probability": ['True']
    },
    {   # linear: only C matters
        "model__kernel": ["linear"],
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    {   # rbf: C and gamma interact
        "model__kernel": ["rbf"],
        "model__C": [0.1, 1, 10, 100, 1000],
        "model__gamma": ["scale", 1e-3, 1e-2, 1e-1],
    },
    {   # poly: degree and gamma matter
        "model__kernel": ["poly"],
        "model__C": [0.1, 1, 10],
        "model__degree": [2, 3, 4],
        "model__gamma": ["scale", 1e-2],
    },
]


sv_search = GridSearchCV(
    param_grid=hp_parameters,
    estimator=model,
    cv = cv,
    n_jobs=-1
)

sv_search.fit(X_train, y_train, groups=groups)
# {'model__C': 10, 'model__gamma': 0.01, 'model__kernel': 'rbf'}

hp_parameters = [{
    "model__kernel": ["rbf"],
    "model__C": [3, 5, 10, 20, 30],
    "model__gamma": [0.003, 0.005, 0.01, 0.02, 0.03],
}]

sv_search = GridSearchCV(model, hp_parameters, cv=cv,
                         scoring="f1_macro", n_jobs=-1)



sv_search.fit(X_train, y_train, groups=groups)
final = clone(sv_search.best_estimator_)
final.set_params(model__probability=True)
final.fit(X_train, y_train)





model_tuned_svc = final.fit(X_train, y_train)
y_pred = final.predict(X_test)
cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
sns.heatmap(cm, annot=True, fmt='d')

f1_score(y_pred=y_pred, y_true=y_test, average='macro')


joblib.dump(model_tuned_svc, 'saved_models/svc_search_tunned.joblib')

print(model_tuned_svc.classes_)