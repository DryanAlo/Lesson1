import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris= load_iris()
test_idx=[0,50,100]

#training data
train_target =np.delete(iris.target,test_idx)
train_data=np.delete(iris.data,test_idx,axis=0)
#data = features
#target = label


#testing data
test_target= iris.target[test_idx]
test_data= iris.data[test_idx]

clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)
 
value= clf.predict([5.1,3.5,1.4,0.2])

if value==0:
	print "It is a setosa"
