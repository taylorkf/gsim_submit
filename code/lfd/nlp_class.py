"""
This file takes a path to the training files and an annotated conllx file.
Ex: python test_groundings ~/pathto/gsim/data ~/pathto/gsim/data/user_num/experiment_num
This file outputs the objects  (in the yaml format) for the moving object and landmark
	object and the number of a location learned by the gmm program
"""

from grounding_all import WordLearner
import re
import os
import yaml

class nlp():
	def __init__(self):
		self.grounding = {}

	def up_tree(self,line_num,lines, moving_objects,groundings):
		tokens = [t.strip() for t in re.findall(r"[\w']+|[.,!?;:]", re.sub(r"[.,!?;:]",' ',lines[line_num-1]))]
		word = tokens[1]
		if word in groundings:
			for o in moving_objects:
				if (o[3] not in groundings[word]) and (o[4] + 5 not in groundings[word]):
					moving_objects.remove(o)
		next = tokens[6]
		if next == "0":
			return
		else:
			return self.up_tree(int(next),lines, moving_objects,groundings)

	def down_tree(self,line_num,lines,landmark_objects, depth):
		if(len(landmark_objects) == 1):
			return landmark_objects
		tokens = [t.strip() for t in re.findall(r"[\w']+|[.,!?;:]", re.sub(r"[,!?;:]",' ',lines[line_num-1]))]
		word = tokens[1]
		hold_landmarks = list(landmark_objects)
		if word in self.groundings and depth != 0:
			in_scenes = False
			for o in landmark_objects:
				if(o[3] in self.groundings[word] or o[4] + 5 in self.groundings[word]):
					in_scenes = True
					break
			if(in_scenes):
				for o in landmark_objects:
					if (o[3] not in self.groundings[word]) and (o[4] + 5 not in self.groundings[word]):
						hold_landmarks.remove(o)
		landmark_objects = list(hold_landmarks)
		for line in lines:
			tokens = [t.strip() for t in re.findall(r"[\w']+|[.,!?;:]", re.sub(r"[!?;:]",' ',line))]
			if(tokens != []):
				if int(tokens[6]) == line_num:
					landmark_objects = self.down_tree(int(tokens[0]),lines,landmark_objects,depth+1)
		return landmark_objects


	def train(self,training_set):
		path_to_files = "../data/"
		learner = WordLearner(path_to_files,training_set)
		learner.ground()
		self.groundings = learner.grounding_dict


	def predict_goal(self,user,exp,num):
		path_to_dependencies = "../data/user_"+str(user)+"/experiment_"+str(exp)+"/"
		landmark_objects = []
		for file_name in os.listdir(path_to_dependencies):
			if file_name == "worlds.yaml":
				with open(path_to_dependencies + "/" + file_name, 'r') as f:
					world_files = yaml.load(f)
					loc = world_files["worlds"][0]
					with open(path_to_dependencies + "/../../" + loc, 'r') as g:
						world = yaml.load(g)
						object_list = world["objects"]
						for o in object_list:
							landmark_objects.append(o)
		spatial_features = [11,12,13,14,15,16,17,18,19]
		spatial_location = None
		spatial_location_fallback = None
		spatial_relation = []
		test = open(path_to_dependencies + "annotations"+str(user)+str(exp)+".conllx", 'r')
		lines = test.readlines()
		for line in lines:
			tokens = [t.strip() for t in re.findall(r"[\w']+|[.,!?;:]", re.sub(r"[.,!?;:]",' ',line))]
			if(tokens != []):
				word = tokens[1].lower()
				if(tokens[7] == "prep" and spatial_location_fallback == None):
					spatial_location_fallback = tokens[0]
				if word in self.groundings:
					if not set(self.groundings[word]).isdisjoint(set(spatial_features)):
						spatial_location = tokens[0]
						spatial_relation = self.groundings[word]
						break
		if(spatial_location != None):
			landmark_objects = self.down_tree(int(spatial_location),lines,landmark_objects,0)
		elif(spatial_location_fallback != None):
			landmark_objects = self.down_tree(int(spatial_location_fallback),lines,landmark_objects,0)
		#print(landmark_objects)
		return([spatial_relation, landmark_objects])





