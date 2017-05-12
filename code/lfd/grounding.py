import yaml
import numpy as np
import matplotlib.pyplot as plt
import lfd

shapes = {'circle':0,'square':1,'triangle':2,'star':3,'diamond':4}
colors = {'blue':0, 'red':1,'green':2,'purple':3,'yellow':4,'orange':5}

id_to_shape = {item[1]:item[0] for item in shapes.items()}
id_to_color = {item[1]:item[0] for item in colors.items()}
id_to_spatial = {5:'up', 7:'on', 0:'right', 2:'left', 1:'down', 3:'other', 4:'other', 6:'other', 8:'other'}

def getDisplacements(user_id, exp_id, demo_id):
    yaml_file = "../data/user_"+str(user_id)+"/experiment_"+str(exp_id)+"/displacements" + str(demo_id) + ".yaml"
    data = lfd.getYamlData(yaml_file)
    return data['objects']

#TODO does it help to restrict the number of worlds per sentence
#TODO figure out regexp for getting rid of punctuation
def getSentenceWorldPairs(user_exp_tuples):
    #assume we use first demo for each user and pair with given sentence
    demo_id = 0
    
    sw_pairs = []
    
    
    for user_id, exp_id in user_exp_tuples:
        #get sentence
        sentence_file = "../data/user_"+str(user_id)+"/experiment_"+str(exp_id)+"/task.txt"
        f = open(sentence_file)
        line = f.readline().strip().replace('-',' ').replace('.',' ').replace(',',' ').replace('?',' ').replace('(',' ').replace(')',' ').replace('\'',' ').lower()

        world = World(user_id, exp_id, demo_id)
        sw_pairs.append((line, world))
    
    return sw_pairs


def getUnigramWorldMapping(user_exp_tuples):
    sw_pairs = getSentenceWorldPairs(user_exp_tuples)
    unigram_map = {}
    for s,w in sw_pairs:
        #add each word in sentence mapping to world
        for word in s.split():
            if word not in unigram_map:
                unigram_map[word] = {w}
            else:
                unigram_map[word].add(w)
    return unigram_map
    
def getUnigramIntersections(unigram_map):
    unigram_groundings = {}
    for unigram in unigram_map:
        world_list = list(unigram_map[unigram])
        invariants = world_list[0]
        for world in world_list:
            invariants = invariants.intersection(world)
        unigram_groundings[unigram] = invariants
    return unigram_groundings

class World(object):

    """initialize with yaml file"""
    def __init__(self, user_id=None, exp_id=None, demo_id=None):
        
        self.shape_set = set()
        self.color_set = set()
        self.displacement_set = set()
        if user_id is not None and exp_id is not None and demo_id is not None:
            yaml_file = lfd.getYamlFile(user_id, exp_id, demo_id)
            #print yaml_file
            #get world data from yaml_file
            data = lfd.getYamlData(yaml_file)
            #print data
            #add objects on table
            for obj in data['objects']:
                self.shape_set.add(obj[3])
                self.color_set.add(obj[4])
            #add task object
            self.shape_set.add(data['task_object'][1])
            self.color_set.add(data['task_object'][2])
            #add displacements
            displacements = getDisplacements(user_id, exp_id, demo_id)
            for d in displacements:
                self.displacement_set.add(d[0])
                
    
    def toEnglish(self):
        eng_str = ""
        if len(self.shape_set) > 0:
            eng_str += "SHAPE: "
            for s in self.shape_set:
                eng_str += id_to_shape[s] + ", "
        if len(self.color_set) > 0:
            eng_str += "COLOR: "
            for c in self.color_set:
                eng_str += id_to_color[c] + ", "
        if len(self.displacement_set) > 0:
            eng_str += "SPATIAL: "
            for d in self.displacement_set:
                eng_str += id_to_spatial[d] + ", "
            
        return eng_str
        
    
    """initialize with sets"""
    def initialize(self, shapes, colors, displacements):
        self.shape_set = shapes
        self.color_set = colors
        self.displacement_set = displacements
               
    def addShape(self, shape_id):
        self.shape_set.add(shape_id)
    
    def addColor(self, color_id):
        self.color_set.add(color_id)
        
    def addDisplacement(self, displacement_id):
        self.displacement_set.add(displacement_id)

    def intersection(self, other):
        intersected = World()
        new_shape_set = self.shape_set.intersection(other.shape_set)
        new_color_set = self.color_set.intersection(other.color_set)
        new_disp_set = self.displacement_set.intersection(other.displacement_set)
        intersected.initialize(new_shape_set, new_color_set, new_disp_set)
        return intersected
        
    def __str__(self):
        return "shapes: " + str(self.shape_set) + ", colors: " + str(self.color_set) + ", displ: " + str(self.displacement_set)

    def __eq__(self, other):
        return other and self.shape_set == other.shape_set and self.color_set == other.color_set and self.displacement_set == other.displacement_set

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(tuple(self.shape_set)) * 31 * hash(tuple(self.color_set)) * 11 * hash(tuple(self.displacement_set))


def learnSimpleGroundings(train_data):
    unigram_map = getUnigramWorldMapping(train_data)
    #find intersections of all worlds associated with each unigram 
    unigram_groundings = getUnigramIntersections(unigram_map)
    
    sorted_unigrams = unigram_groundings.keys()
    sorted_unigrams.sort()
    
    #print out all unigrams in alphabetical order
#    for u in sorted_unigrams:
#        print u + " = " + unigram_groundings[u].toEnglish()
#    
    
    
    #print out unigrams with one grounding that had more than min_occur occurances of word 
    print "--------MOST LIKELY-------------"
    min_occur = 10
    
    for u in sorted_unigrams:
        if len(unigram_groundings[u].shape_set) + len(unigram_groundings[u].color_set) + len(unigram_groundings[u].displacement_set) == 1 and len(unigram_map[u]) >= min_occur:
            print u + " = " + unigram_groundings[u].toEnglish()


#TODO continue to work on this
def score(unigram, world, unigram_map):
    freq_unigram = len(unigram_map[unigram])

    freq_not_unigram = 0
    for u in unigram_map:
        if u != unigram:
            freq_not_unigram += len(unigram_map[u])
    
    freq_world_unigram = 0
    for w in unigram_map[unigram]:
        w_intersection = world.intersection(w)
        if len(w_intersection.shape_set) + len(w_intersection.color_set) + len(w_intersection.displacement_set) > 0:
            freq_world_unigram += 1
    
    freq_world = 0
    for u in unigram_map:
        for w in unigram_map[unigram]:
            w_intersection = world.intersection(w)
            if len(w_intersection.shape_set) + len(w_intersection.color_set) + len(w_intersection.displacement_set) > 0:
                freq_world += 1

    freq_world_not_unigram = freq_world - freq_world_unigram

    return float(freq_world_unigram) / freq_unigram - float(freq_world_not_unigram) / freq_not_unigram
    
    
def getLexicon(unigram_map, threshold):
    lexicon = {}
    for unigram in unigram_map:
        print unigram
        meanings = set()
        size = 0     
        new_size = 1
        while size != new_size:
            size = new_size
            #add intersections for every pair of worlds
            for world1 in unigram_map[unigram]:
                for world2 in unigram_map[unigram]:
                    meanings.add(world1.intersection(world2))
            new_size = len(meanings)
            #print new_size
        #add all entries in meanings with scores higher than threshold
        meanings_list = list(meanings)
        scores = [score(unigram, m, unigram_map) for m in meanings_list]
        #sort from highest to lowest
        sorted_indices = list(np.argsort(scores))
        sorted_indices.reverse()
        likely_meanings = []
        for i in range(len(scores)):
            if scores[sorted_indices[i]] > threshold:
                likely_meanings.append(meanings_list[sorted_indices[i]])
        
        lexicon[unigram] = likely_meanings
        
    return lexicon


def main():
    #first learn mixtures and write mu and cov dicts to yaml file 
    train_data = [(u,e) for u in range(30) for e in range(0,4)]
    
    #test out world class
#    world1 = World()
#    world2 = World()
#    
#    world1.initialize(set([1,2,3]), set([1,2,3]), set([1,2,3]))
#    world2.initialize(set([2,3,4]), set([5,6,7]), set([3,4,5]))
#    world3 = world1.intersection(world2)
#    print world1
#    print world2
#    print world3
#    
#    world4 = World(0,0,0)
#    print world4
#    print world4.intersection(world3)
#    print world4.toEnglish()

    #test getting sentence world pairs
#    sw_pairs = getSentenceWorldPairs(train_data)
#    s,w = sw_pairs[0]
#    print s
#    print w
    
    ##Learn simple groundings
    #learnSimpleGroundings(train_data)
    

    ####Chen and mooney grounding   
    threshold = 0.4
    unigram_map = getUnigramWorldMapping(train_data)
    #find intersections of all worlds associated with each unigram 
    unigram_groundings = getLexicon(unigram_map, threshold)
    sorted_unigrams = unigram_groundings.keys()
    sorted_unigrams.sort()
    
    #print out all unigrams in alphabetical order
#    for u in sorted_unigrams:
#        print u + " = " + unigram_groundings[u].toEnglish()
#    
    
    
    #print out unigrams with one grounding that had more than min_occur occurances of word 
    print "--------MOST LIKELY-------------"
    
    for u in sorted_unigrams:
        if len(unigram_groundings[u]) > 0:
            print u + " = " + unigram_groundings[u][0].toEnglish()

if __name__=="__main__":
    main()
