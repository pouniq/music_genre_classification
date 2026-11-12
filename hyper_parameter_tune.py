import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

import joblib

df = pd.read_csv('gtzan_5s_features.csv')

groups = df['file']
len(groups)

X = df.drop(columns=["file", "genre", "segment", "start_sec"])
y = df['genre']

y.unique()
y = y.map({
    'blues': 1,
    'classical':2,
    'country': 3,
    'disco': 4,
    'hiphop': 5,
    'jazz': 6,
    'metal': 7, 
    'pop': 8, 
    'reggae': 9,  
    'rock': 10
})


param_grid = {
    'learning_rate': [0.05,0.1, 0.2],
    'max_leaf_nodes': [15, 30, 60],
    'l2_regularization': [0.0, 0.5, 1.0]
}


gf = GroupKFold(n_splits=5)
gf_inner = GroupKFold(n_splits=3)

y_preds = {}
scores_f1 = []
scores_acc = []

best_params = []
for split_num ,(train_idx, test_idx) in enumerate(gf.split(X, y, groups=groups)):
    
    print('Training Index', train_idx, 'Test Index', test_idx)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    current_group = groups[train_idx]
    

    search_grid = GridSearchCV(
        HistGradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        cv = gf_inner,
        scoring='accuracy',
        n_jobs=-1
    )
    
    search_grid.fit(X_train, np.ravel(y_train), groups=current_group)
    y_pred = search_grid.predict(X_test)
    best_params.append(search_grid.best_params_)
    
    y_preds[split_num] = {"test_idx": test_idx, "y_true": np.ravel(y_test), "y_pred": y_pred}
    scores_f1.append(f1_score(y_test, y_pred, average="macro"))
    scores_acc.append(accuracy_score(y_test, y_pred))
    print(split_num + 1, search_grid.best_params_, round(scores_acc[-1], 3))
    
    
    # model = HistGradientBoostingClassifier()
    # y_train = np.ravel(y_train)
    # model = model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # y_preds[split_num] = {
    #     "test_idx": test_idx,
    #     "y_true": np.ravel(y_test),
    #     "y_pred": model.predict(X_test)
    # }
    # scores_f1.append(f1_score(y_test, y_pred, average="macro"))
    # scores_acc.append(accuracy_score(y_test, y_pred=y_pred))

print(f'f1 score mean: {np.mean(scores_f1), np.std(scores_f1)}')
print(f'accuracy score mean: {np.mean(scores_acc), np.std(scores_acc)}')

scores_acc
scores_f1 


best_params


## best params -> l2: 0, lr: 0.2, mln: 15

best_model = HistGradientBoostingClassifier(l2_regularization=0, learning_rate=0.2, max_leaf_nodes=15)
model = best_model.fit(X_train, y_train)
y_pred = model.predict(X_test)


confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred=y_pred)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)


plt.figure(figsize=(10, 7))
plt.plot(range(1, 6), scores_acc, marker="o")
plt.xlim(0.5, len(scores_acc) + 0.5)
plt.xticks(range(1, len(scores_acc) + 1))
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.show()


# save the tuned model

best_model = clone(best_model).fit(X_train, y_train)
joblib.dump(best_model, "saved_models/tuned_best_model.joblib")



joblib.dump(y, "saved_models/tunned_genre_names.joblib")
