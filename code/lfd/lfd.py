import yaml
import numpy as np
import matplotlib.pyplot as plt
import random


def getRandomPointOnTable(user_id, exp_id, n_demos):
    yaml_file = getYamlFile(user_id, exp_id, 0)
    x,y,w,h = getTableTopDimensions(yaml_file)
    return [random.randint(x,x+w), random.randint(y,y+h)]
    

def getTableTopDimensions(yaml_file):
    data = getYamlData(yaml_file)
    #get table: [100, 100, 800]
    table = data['table']
    return table[0], table[1], table[2], table[2]

def getYamlFiles(user_id, experiment_id, num):
    #get worlds.yaml file
    yaml_meta_filename = "../data/user_"+str(user_id)+"/experiment_"+str(experiment_id)+"/worlds.yaml"
    #print yaml_meta_filename
    meta_files = getYamlData(yaml_meta_filename)
    return meta_files['worlds'][0:num]

def getYamlFile(user_id, experiment_id, demo_num):
    #get worlds.yaml file
    yaml_meta_filename = "../data/user_"+str(user_id)+"/experiment_"+str(experiment_id)+"/worlds.yaml"
    #print yaml_meta_filename
    meta_files = getYamlData(yaml_meta_filename)
    return meta_files['worlds'][demo_num]


def getFeatureCoordinates(shape_color, yaml_file):
    data = getYamlData(yaml_file)
    objects = data['objects']
    obj_coord = []
    for obj in objects:
        obj_features = (obj[3],obj[4])
        if obj_features == shape_color:
            return [obj[0], obj[1]]
    print ("Couldn't find the object specified by the shape_color feature tuple")
    return 

"""hard coded to get all user info except for last experiment
    this corresponds to 80% train 20% test
"""  
def getYamlTrainFiles():
    num_users = 30
    exp_per_user = 5
    all_files = []
    for user_id in range(num_users):
        for exp_id in range(exp_per_user - 1):
            all_files.extend(getYamlFiles(user_id, exp_id))
    return all_files
     

"""get (x,y) of target for a certain yaml file"""
def getTargetCoordinates(yaml_file):
    data = getYamlData(yaml_file)
    return data['target'][0], data['target'][1]   

"""get the last position from the motion data corresponding to yaml file"""
def getPlacementCoordinates(yaml_file, user_id, exp_id):
    #print yaml_file
    filename = yaml_file.split("/")[-1]
    #print filename
    rel_id_exp = filename.split(".")[0] # get the part right before .yaml
    data = rel_id_exp.split("_")
    demo_id = int(data[-1])
    motion_file = "../data/user_" + str(user_id) + "/experiment_" + str(exp_id) + "/motion.npy"
    motion_data = np.load(motion_file)

    return [float(motion_data[demo_id][-1][0]), float(motion_data[demo_id][-1][1])]
    

def getMeanPlacementCoordinates(user_id, exp_id, num_demos):
    yaml_files = getYamlFiles(user_id, exp_id, num_demos)
    all_placements = []
    for yaml_f in yaml_files:
        #print yaml_f
        placement = getPlacementCoordinates(yaml_f, user_id, exp_id)
        all_placements.append(np.array(placement))

    return np.mean(np.array(all_placements),0)
        

"""get (x,y) for each object in world"""
def getObjectCoordinates(yaml_file):
    data = getYamlData(yaml_file)
    objects = data['objects']
    obj_coord = []
    for obj in objects:
        #add coordinats of obj to list
        obj_coord.append((obj[0], obj[1]))

    return obj_coord
   
"""Compute displacements for a single yaml file as dictionay between features and displacements"""
def computeDisplacements(yaml_file, user_id, experiment_id):
    #find target and object coords
    data = getYamlData(yaml_file)
    target_coord = getPlacementCoordinates(yaml_file, user_id, experiment_id)
    objects = data['objects']
    obj_coord = []
    displacements = {}
    for obj in objects:
        #add coordinates of obj to list associated with features
        obj_features = (obj[3],obj[4])
        displacements[obj_features] = np.array(target_coord) - np.array([float(obj[0]), float(obj[1])])
    return displacements

"""returns ordered list of displacements for a single world"""
def getDisplacementList(yaml_file, user_id, experiment_id):
    #find target and object coords
    data = getYamlData(yaml_file)
    target_coord = getPlacementCoordinates(yaml_file, user_id, experiment_id)
    objects = data['objects']
    obj_coord = []
    displacements = []
    for obj in objects:
        #add coordinates of obj to list associated with features
        displacements.append(np.array(target_coord) - np.array([float(obj[0]), float(obj[1])]))
    return displacements

def getAllDisplacements(user_id, experiment_id, num_demos):
    yaml_files = getYamlFiles(user_id, experiment_id, num_demos)
    all_disp = {}
    for yaml_f in yaml_files:
        #print yaml_f
        displ = computeDisplacements(yaml_f, user_id, experiment_id)   
        for d in displ:
            #print d
            if d not in all_disp:
                all_disp[d] = [displ[d]]
            else:
                all_disp[d].append(displ[d])
            #print all_disp
    #convert all to 2-d numpy arrays
    for f in all_disp:
        all_disp[f] = np.array(all_disp[f])
    return all_disp
         
"""takes a dictionary of feature keys with 2-d arrays of
   displacements per feature with each row a displacement 
   and plots
"""
def plotDisplacements(user_id, experiment_id, num_demos):
    #grab files    
    world_files = getYamlFiles(user_id, experiment_id, num_demos)
    #print world_files
    #grab coordinates
    displacement_dict = getAllDisplacements(user_id, experiment_id, num_demos)
    #print "displacements", all_disp

#    tot_vars = getTotalVariances(all_disp)  
#    for feature in tot_vars:
#        print feature, tot_vars[feature]
#    
#    disp_mus = getMeanDisplacements(all_disp)  
#    for feature in disp_mus:
#        print feature, disp_mus[feature]
    plt.figure()
    colors = ['r','b','g']
    cnt = 0
    feature_list = [f for f in displacement_dict]
    for feature in feature_list:
        displace_array = displacement_dict[feature]
        plt.plot(displace_array[:,0],displace_array[:,1],colors[cnt]+'o',label=feature)
        cnt += 1
    mus = getMeanDisplacements(displacement_dict)
    print (mus)
    cnt = 0
    for f in feature_list:
        plt.plot(mus[f][0], mus[f][1], colors[cnt]+'x')
        cnt += 1
    
    plt.legend(loc='best')
    plt.show()

"""takes a 2-d array with each row a displacement and plots"""    
def plotDisplacementsRaw(disp_array):
   plt.figure()
   plt.plot(disp_array[:,0],disp_array[:,1],'o')
   plt.show() 

def getTotalVariances(displacement_dict):
    displ_var = {}
    for feature in displacement_dict:
        displace_array = np.array(displacement_dict[feature])
        #print displace_array.T
        displ_var[feature] = np.trace(np.cov(displace_array.T))
    return displ_var
        
def getMeanDisplacements(displacement_dict):
    displ_var = {}
    for feature in displacement_dict:
        displace_array = np.array(displacement_dict[feature])
        #print displace_array.T
        displ_var[feature] = np.mean(displace_array,0)
    return displ_var


"""given a landmark (shape,color) find average displacement from this landmark over the demos"""
def getDisplacementFromLandmark(user_id, experiment_id, num_demos, landmark):
    all_disp = getAllDisplacements(user_id, experiment_id, num_demos)
    disp_mus = getMeanDisplacements(all_disp)
    return disp_mus[landmark]

"""returns shape-color tuple and mean displacement vector"""
def getMostLikelyLandmarkDisplacement(user_id, experiment_id, num_demos):
    
    if num_demos == 1:
        #just pick randomly since all have zero variance
        all_disp = getAllDisplacements(user_id, experiment_id, num_demos)
        fkeys = [f for f in all_disp]
        landmark = fkeys[np.random.randint(len(fkeys))]
        return landmark, all_disp[landmark]
    else:
        #calc displacements
        all_disp = getAllDisplacements(user_id, experiment_id, num_demos)
        tot_vars = getTotalVariances(all_disp)  
        min_var = float("inf")
        for feature in tot_vars:
            if tot_vars[feature] < min_var:
                landmark = feature
                min_var = tot_vars[feature]
        disp_mus = getMeanDisplacements(all_disp)  

        return landmark, disp_mus[landmark]

def getYamlData(yaml_filename):
    #how to open a yaml file
    with open(yaml_filename, 'r') as stream:
        try:
            world = yaml.load(stream)
            #print world
        except yaml.YAMLError as exc:
            print (exc )
            sys.exit()
    return world
    
"""return all displacements from all files in world_files
   in one big 2-d array with each row a displacement [x,y]
"""
def getAllDisplacementsConcat(user_exp_tuples, num_demos):
    disp = []
    for user_id, exp_id in user_exp_tuples:
        disp_dict = getAllDisplacements(user_id, exp_id, num_demos)
        for feature in disp_dict:
            disp.extend(disp_dict[feature])
    return np.array(disp)
    


def main():
    print ("grabbing files")
    #need to get a set of files for demonstrations
    #specfify user and experiment
    user_id = 0        #between 0 and 29
    experiment_id = 0  #between 0 and 4
    num_demos = 10
    
    shape_color, displacement = getMostLikelyLandmarkDisplacement(user_id, experiment_id, num_demos)
    print ("most likely landmark", shape_color)
    print ("displacement", displacement)
    plotDisplacements(user_id, experiment_id, num_demos)  


    user_ids = range(30)
    exp_ids = range(4)  #shouldn't include last for each user
    user_exp_tuples = [(u,e) for u in user_ids for e in exp_ids]

    train_disp = getAllDisplacementsConcat(user_exp_tuples, num_demos)
    plotDisplacementsRaw(train_disp)
    




if __name__=="__main__":
    main()
