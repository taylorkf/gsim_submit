import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from nlp_class import nlp
import random

import lfd
import gmm
import nlp_lfd




"""return the average error from randomly guessing a position within the tabletop"""
def averageDistanceToPointError(placement, user_id, exp_id, max_demos, total_demos):
    sum_errors = 0.0
    #for each demo find location of shape_color
    for demo_id in range(max_demos, total_demos):
        yaml_file = lfd.getYamlFile(user_id, exp_id, demo_id)
        true_coord = lfd.getPlacementCoordinates(yaml_file, user_id, exp_id)
        #print np.array(landmark_coord)
        #print np.array(displacement)
        #add displacement
        pred_coord = np.array(placement)

        #compare displacement to actual human placement
        sum_errors += np.linalg.norm(pred_coord - true_coord)
    return sum_errors / float(total_demos - max_demos)  


"""compute the average distance error over worlds specified by user_id, exp_id, max_demos:total_demos
"""
def averageDistanceError(shape_color, displacement, user_id, exp_id, max_demos, total_demos):
    sum_errors = 0.0
    #for each demo find location of shape_color
    for demo_id in range(max_demos, total_demos):
        yaml_file = lfd.getYamlFile(user_id, exp_id, demo_id)
        landmark_coord = lfd.getFeatureCoordinates(shape_color, yaml_file)
        true_coord = lfd.getPlacementCoordinates(yaml_file, user_id, exp_id)
        #print np.array(landmark_coord)
        #print np.array(displacement)
        #add displacement
        pred_coord = np.array(landmark_coord) + np.array(displacement)

        #compare displacement to actual human placement
        sum_errors += np.linalg.norm(pred_coord - true_coord)
    return sum_errors / float(total_demos - max_demos)  

    
"""return NLP+LfD prediction
   if relation is empty or objects has more than one entry, just uses pure LfD
"""
def getNLP_LfD_LandmarkDisplacement(nlp_grounder, user_id, exp_id, n_demos):
    relation_offset = 11  #hard coded offset to get back to relations in range 0:8
    relation, object_info = nlp_grounder.predict_goal(user_id, exp_id, n_demos)
    print (len(relation), len(object_info))
    print (relation)
    #get lfd to use along with language or in place of language if ambiguous
    lfd_shape_color, lfd_displacement = lfd.getMostLikelyLandmarkDisplacement(user_id, exp_id, n_demos)
    
    #check if we have one relation and one object
    if len(relation) == 1 and len(object_info) == 1:
        
        #print "grounding both"
        #first check if nlp grounded landmark is actually a possible landmark
        nlp_shape_color = (object_info[0][3], object_info[0][4])
        yaml_file = lfd.getYamlFile(user_id, exp_id, 0)
        landmark_coord = lfd.getFeatureCoordinates(nlp_shape_color, yaml_file)
        if landmark_coord is None:
            
            #print "NLP returned invalid landmark"
            return lfd_shape_color, lfd_displacement
        else: #use valid landmark and displacement to guess placement location
            
            #get shape,color
            nlp_shape_color = (object_info[0][3], object_info[0][4])
            #get displacement from learned gmm
            gmm_filename = "../data/gmm_params.yaml"
            model_data = lfd.getYamlData(gmm_filename)
            nlp_displacement = model_data['mu'][relation[0] - relation_offset]
            return nlp_shape_color, nlp_displacement
    #check if we have only one object and can ground on that
    elif len(object_info) == 1:
        #print "grounding on object only"
        nlp_shape_color = (object_info[0][3], object_info[0][4])
        nlp_displacement = lfd.getDisplacementFromLandmark(user_id, exp_id, n_demos, nlp_shape_color)
        return nlp_shape_color, nlp_displacement
    #if we have one relationship and multiple items, figure out which item by using relationship
    elif len(relation) == 1 and len(object_info)>0:
        #print "grounding on relation only"
        #get predicted displacement based on relationship
        gmm_filename = "../data/gmm_params.yaml"
        model_data = lfd.getYamlData(gmm_filename)
        nlp_displacement = model_data['mu'][relation[0] - relation_offset]
        #get placement data and object locations
        #TODO which demo should I pick from, what should n_demos be??
        yaml_file = lfd.getYamlFile(user_id, exp_id, n_demos)
        placement_coord = lfd.getPlacementCoordinates(yaml_file, user_id, exp_id)
        min_dist = 10000
        for obj in object_info:
            shape_color = (obj[3], obj[4])
            landmark_coord = lfd.getFeatureCoordinates(shape_color, yaml_file)
            predicted_placement = np.array(landmark_coord) + np.array(nlp_displacement)
            placement_error = np.linalg.norm(predicted_placement - placement_coord)
            if placement_error < min_dist:
                min_dist = placement_error
                best_landmark = shape_color
        return best_landmark, nlp_displacement
    #TODO case where only relationship is grounded?
    #elif len(relation) == 1:                    
    #fall back on pure lfd as a last resort
    else:
        
        #print "ambiguous grounding"
        return lfd_shape_color, lfd_displacement
    

"""double check the grounded language against the actual demonstrations to see if it makes sense"""
def getNLP_LfD_LandmarkDisplacementDoubleCheck(nlp_grounder, user_id, exp_id, n_demos, thresh):
    relation_offset = 11  #hard coded offset to get back to relations in range 0:8
    relation, object_info = nlp_grounder.predict_goal(user_id, exp_id, n_demos)
    #print len(relation), len(object_info)
    #print relation
    #get lfd to use along with language or in place of language if ambiguous
    lfd_shape_color, lfd_displacement = lfd.getMostLikelyLandmarkDisplacement(user_id, exp_id, n_demos)
    
    #check if we have one relation and one object
    grounded_prediction = False #flag to check if we think we've found a grounding
    if len(relation) == 1 and len(object_info) == 1:
        
        #print "grounding both"
        #first check if nlp grounded landmark is actually a possible landmark
        nlp_shape_color = (object_info[0][3], object_info[0][4])
        yaml_file = lfd.getYamlFile(user_id, exp_id, 0)
        landmark_coord = lfd.getFeatureCoordinates(nlp_shape_color, yaml_file)
        if landmark_coord is not None:
        
            #get shape,color
            nlp_shape_color = (object_info[0][3], object_info[0][4])
            #get displacement from learned gmm
            gmm_filename = "../data/gmm_params.yaml"
            model_data = lfd.getYamlData(gmm_filename)
            nlp_displacement = model_data['mu'][relation[0] - relation_offset]
            grounded_prediction = True

    #check if we have only one object and can ground on that
    elif len(object_info) == 1:
        #print "grounding on object only"
        nlp_shape_color = (object_info[0][3], object_info[0][4])
        nlp_displacement = lfd.getDisplacementFromLandmark(user_id, exp_id, n_demos, nlp_shape_color)
        grounded_prediction = True

    #if we have one relationship and multiple items, figure out which item by using relationship
    elif len(relation) == 1 and len(object_info)>0:
        #print "grounding on relation only"
        #get predicted displacement based on relationship
        gmm_filename = "../data/gmm_params.yaml"
        model_data = lfd.getYamlData(gmm_filename)
        nlp_displacement = model_data['mu'][relation[0] - relation_offset]
        #get placement data and object locations
        #TODO which demo should I pick from, what should n_demos be??
        yaml_file = lfd.getYamlFile(user_id, exp_id, n_demos)
        placement_coord = lfd.getPlacementCoordinates(yaml_file, user_id, exp_id)
        min_dist = 10000
        for obj in object_info:
            shape_color = (obj[3], obj[4])
            landmark_coord = lfd.getFeatureCoordinates(shape_color, yaml_file)
            predicted_placement = np.array(landmark_coord) + np.array(nlp_displacement)
            placement_error = np.linalg.norm(predicted_placement - placement_coord)
            if placement_error < min_dist:
                min_dist = placement_error
                best_landmark = shape_color
        nlp_shape_color = best_landmark
        grounded_prediction = True

   #TODO case where only relationship is grounded?
    #elif len(relation) == 1:                    
    
    #see if I have a grounded prediction or if I should go with pure lfd
    if grounded_prediction:
        #check if grounding prediction seems to match what's been demonstrated
        ave_error = averageDistanceError(nlp_shape_color, nlp_displacement, user_id, exp_id, 0, n_demos+1)
        #print "ave error", ave_error
        if ave_error < thresh:
            #print "confident in grounding"
            return nlp_shape_color, nlp_displacement, relation, object_info
    #fall back on pure lfd as a last resort
    #print "ambiguous grounding"
    return lfd_shape_color, lfd_displacement, relation, object_info

