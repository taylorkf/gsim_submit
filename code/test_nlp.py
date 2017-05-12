from nlp_class import nlp

"""This file is an example of how to call the nlp class"""


train_set = []
for i in range(30):
	for j in range(5):
		if(i%2!=0 or j != 0) and (i != 4 and i != 5):
			train_set.append((i,j))

nlp = nlp()
nlp.train(train_set)

(relation,shape) = nlp.predict_goal(4,0,5)
shapes = ["circle","square","triangle","star","diamond"]
colors = ["blue","red","green","purple","yellow","orange"]
print("Spatial relation:", relation)
print("Shape:",colors[shape[0][4]],shapes[shape[0][3]])
