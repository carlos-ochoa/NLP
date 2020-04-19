from sklearn.linear_model import LinearRegression
from pickle import load

def load_structure(filename):
    file = open(filename,'rb')
    structure = load(file)
    file.close()
    return structure

# Load data previously saved from scratch implementation
X, y, X_test, y_test = load_structure('training.pkl'),load_structure('y_tr.pkl').T \
                    ,load_structure('testing.pkl'),load_structure('y_te.pkl').T
print(X.shape)
print(y.shape)
lr = LinearRegression()
lr.fit(X,y)
predictions = lr.predict(X_test)
score = lr.score(X,y)
test_score = lr.score(X_test,y_test)
print('Training score: ', score)
print('Testing score: ', test_score)
print('Coefficients:', lr.coef_)

i = 0
for p in predictions:
    print('Expected: ',y_test[i], ' Predicted: ', p)
    i += 1
