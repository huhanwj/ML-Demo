from sklearn.naive_bayes import GaussianNB
# get training data and test data
gnb = GaussianNB()
model = gnb.fit(X_train, y_train)
y_predict = model.predict(X_test)
