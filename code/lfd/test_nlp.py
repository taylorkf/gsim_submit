from nlp_class import nlp

"""This file is an example of how to call the nlp class"""


train_set = []
for i in range(30):
	for j in range(5):
		if(i%2!=0 or j != 0) and (i != 0 and i != 1):
			train_set.append((i,j))

nlp = nlp()
nlp.train(train_set)

spatial_relations = ["right","none","down","left","none","none","up","none","on","none","none"]
shapes = ["circle","square","triangle","star","diamond"]
colors = ["blue","red","green","purple","yellow","orange"]


(relation,shape) = nlp.predict_goal(1,0,1)
print(relation)
print(shape)
print(colors[shape[0][4]])
print(shapes[shape[0][3]])
