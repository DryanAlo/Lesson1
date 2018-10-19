from sklearn import tree
#1 naranja, 2 manzana
#1 rugosa, 2 lisa
#         [[Masa(g),Textura]]

features= [[150,1],[170,1],[140,2],[130,2]]
#Características
labels= [1,1,2,2]
#Resultados

clf= tree.DecisionTreeClassifier()

clf=clf.fit(features,labels)

print clf.predict([[150,1]])
