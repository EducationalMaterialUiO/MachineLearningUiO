"""Chapter 7: listing 6, from the section on xgboost extreme gradient boosting.

Extracted from doc/BookML/chapter7.tex.
"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Classification. reg_lambda is the lambda of Eq. (7.xgbpenalty) and
# gamma the per-leaf cost of Eq. (7.xgbgain).
clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                        reg_lambda=1.0, gamma=0.0, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="logloss")
clf.fit(X_train, y_train)
print("accuracy:", accuracy_score(y_test, clf.predict(X_test)))

# Regression, with early stopping on a validation set
reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                       early_stopping_rounds=20)
reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("best iteration:", reg.best_iteration)
