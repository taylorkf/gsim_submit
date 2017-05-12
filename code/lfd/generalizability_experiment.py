import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from nlp_class import nlp

import lfd
import gmm
import nlp_lfd

from sklearn.model_selection import KFold
import warnings

"""
This is a simple experiment setup for testing nlp+lfd with lfd.
"""
def kFold(data,num_folds):
    max_demos = 4     #maximum number of demos given to robot to learn from (the rest are used to test generalizabiltiy)
    kf = KFold(n_splits=num_folds)
    #for all folds
    """all_lfd_errors = []
    all_nlp_errors = []
    all_random_errors = []
    all_aveplace_errors = []
    all_lfd_object_correct = []
    all_lfd_spatial_correct = []
    all_lfd_both_correct = []
    all_nlp_object_correct = []
    all_nlp_spatial_correct = []
    all_nlp_both_correct = []
    all_nlp_lfd_object_correct = []
    all_nlp_lfd_spatial_correct = []
    all_nlp_lfd_both_correct = []"""
    all_lfd_correct = []
    all_nlp_correct = []
    all_nlp_lfd_correct = []
    for i in range(num_folds):
        all_lfd_correct.append([])
        all_nlp_correct.append([])
        all_nlp_lfd_correct.append([])
    count = 0
    for train,test in kf.split(data):#kf:
        count = count + 1
        print("Cross validation round " + str(count))
        train_data =[]
        test_data = []
        for i in train:
            train_data = train_data + data[i]
        for i in test:
            test_data = test_data + data[i]

        (lfd_correct,nlp_correct, nlp_lfd_correct) = experiment(train_data,test_data)


        """all_lfd_errors.append(list(lfd_errors))
        all_nlp_errors.append(list(nlp_errors))
        all_random_errors.append(list(random_errors))
        all_aveplace_errors.append(list(aveplace_errors))
        all_lfd_object_correct.append(lfd_correct[0])
        all_lfd_spatial_correct.append(lfd_correct[1])
        all_lfd_both_correct.append(lfd_correct[2])
        all_nlp_object_correct.append(nlp_correct[0])
        all_nlp_spatial_correct.append(nlp_correct[1])
        all_nlp_both_correct.append(nlp_correct[2])
        all_nlp_lfd_object_correct.append(nlp_lfd_correct[0])"""
        for i in range(3):
            all_lfd_correct[i].append(lfd_correct[i])
            all_nlp_correct[i].append(nlp_correct[i])
            all_nlp_lfd_correct[i].append(nlp_lfd_correct[i])
        
    ###################
    #plot errors
    ###################
    """lfd_errors = np.zeros((len(test_data), max_demos))  
    nlp_errors = np.zeros((len(test_data), max_demos)) 
    random_errors = np.zeros((len(test_data), max_demos))  
    aveplace_errors = np.zeros((len(test_data), max_demos)) 
    lfd_spatial_correct = np.zeros(max_demos)
    lfd_object_correct = np.zeros(max_demos)
    lfd_both_correct = np.zeros(max_demos)
    nlp_spatial_correct = np.zeros(max_demos)
    nlp_object_correct = np.zeros(max_demos)
    nlp_both_correct = np.zeros(max_demos)"""

    
    plotCorrect(all_lfd_correct,all_nlp_correct,all_nlp_lfd_correct)
    
    #plot(lfd_errors,nlp_errors,random_errors,aveplace_errors)
    
def plotCorrect(lfd_correct,nlp_correct,nlp_lfd_correct):
    max_demos = 4
    for i in range(3):
        lfd_correct[i] = np.divide(lfd_correct[i],27)
        nlp_correct[i] = np.divide(nlp_correct[i],27)
        nlp_lfd_correct[i] = np.divide(nlp_lfd_correct[i],27)
    
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0,1])
    
    plt.errorbar(range(1,max_demos+1),np.mean(lfd_correct[0],0), yerr=np.std(lfd_correct[0],0), fmt='bo-', label='lfd')
    plt.errorbar(range(1,max_demos+1),np.mean(nlp_correct[0],0), yerr=np.std(nlp_correct[0],0), fmt='g^--', label='nlp')
    plt.errorbar(range(1,max_demos+1),np.mean(nlp_lfd_correct[0],0), yerr=np.std(nlp_lfd_correct[0],0), fmt='r*-.', label='nlp+lfd')
    plt.xticks(range(1,max_demos+1))
    plt.xlabel('number of demonstrations')
    plt.ylabel('percent of landmarks correct')
    plt.legend(loc='best')
    plt.show()
    plt.clf()
    
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.errorbar(range(1,max_demos+1),np.mean(lfd_correct[1],0), yerr=np.std(lfd_correct[1],0), fmt='bo-', label='lfd')
    plt.errorbar(range(1,max_demos+1),np.mean(nlp_correct[1],0), yerr=np.std(nlp_correct[1],0), fmt='g^--', label='nlp')
    plt.errorbar(range(1,max_demos+1),np.mean(nlp_lfd_correct[1],0), yerr=np.std(nlp_lfd_correct[1],0), fmt='r*-.', label='nlp+lfd')
    plt.xticks(range(1,max_demos+1))
    plt.xlabel('number of demonstrations')
    plt.ylabel('percent of spatial relations correct')
    plt.legend(loc='best')
    plt.show()
    plt.clf()
    
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.errorbar(range(1,max_demos+1),np.mean(lfd_correct[2],0), yerr=np.std(lfd_correct[2],0), fmt='bo-', label='lfd')
    plt.errorbar(range(1,max_demos+1),np.mean(nlp_correct[2],0), yerr=np.std(nlp_correct[2],0), fmt='g^--', label='nlp')
    plt.errorbar(range(1,max_demos+1),np.mean(nlp_lfd_correct[2],0), yerr=np.std(nlp_lfd_correct[2],0), fmt='r*-.', label='nlp+lfd')
    plt.xticks(range(1,max_demos+1))
    plt.xlabel('number of demonstrations')
    plt.ylabel('percent of landmarks/spatial relations correct')
    plt.legend(loc='best')
    plt.show()

def experiment(train_data,test_data):
    total_demos = 10  #total number of demos possible
    max_demos = 4     #maximum number of demos given to robot to learn from (the rest are used to test generalizabiltiy)
    thresh = 150
    #TODO learn the gmm componenets and label the data
    print ("--learning gmm components--")
    gmm_model = gmm.labelTrainingDataDisplacements(train_data, total_demos)
    gmm.labelTestingDataDisplacements(test_data, total_demos, gmm_model)
    #learn groundings with training data
    print ("--grounding language--")
    nlp_grounder = nlp()
    nlp_grounder.train(train_data)

    ######################
    #testing code
    ######################
    print ("--testing generalization error--")
    #matrix of zeros where each row is new test case and cols are ave errors for learning from 1:max_demos demonstrations
    lfd_errors = np.zeros((len(test_data), max_demos))  
    nlp_errors = np.zeros((len(test_data), max_demos)) 
    random_errors = np.zeros((len(test_data), max_demos)) 
    aveplace_errors = np.zeros((len(test_data), max_demos))
    lfd_correct = np.zeros((3,max_demos))
    nlp_correct = np.zeros((3,max_demos))
    nlp_lfd_correct = np.zeros((3,max_demos))
    """lfd_spatial_correct = np.zeros(max_demos)
    lfd_object_correct = np.zeros(max_demos)
    lfd_both_correct = np.zeros(max_demos)
    nlp_spatial_correct = np.zeros(max_demos)
    nlp_object_correct = np.zeros(max_demos)
    nlp_both_correct = np.zeros(max_demos)
    nlp_lfd_spatial_correct = np.zeros(max_demos)
    nlp_lfd_object_correct = np.zeros(max_demos)
    nlp_lfd_both_correct = np.zeros(max_demos)"""
    
    for i in range(len(test_data)):
        user_id, exp_id = test_data[i]
        for n_demos in range(1,max_demos+1):
            #get best guess of landmark and displacement using pure LfD
            lfd_shape_color, lfd_displacement = lfd.getMostLikelyLandmarkDisplacement(user_id, exp_id, n_demos)
            #guess landmark and displacement using NLP+LfD
            nlp_lfd_shape_color, nlp_displacement, nlp_relation, nlp_shape_color = nlp_lfd.getNLP_LfD_LandmarkDisplacementDoubleCheck(nlp_grounder, user_id, exp_id, n_demos, thresh)
            world = lfd.getYamlFile(user_id, exp_id,0)
            
            lfd_relation = gmm_model.predict(lfd_displacement)[0]
            nlp_lfd_relation = gmm_model.predict(nlp_displacement)[0]

            with open(world, 'r') as f:
                world_file = yaml.load(f)
                landmark =  world_file["objects"][0]
                with open("../data/user_"+str(user_id)+"/experiment_"+str(exp_id)+"/displacements0.yaml",'r') as g:
                    disp_file = yaml.load(g)
                    disp = disp_file["objects"][0]
                if len(nlp_shape_color) == 1 and nlp_shape_color[0][3] == landmark[3] and nlp_shape_color[0][4] == landmark[4]:
                    nlp_correct[0][n_demos-1] = nlp_correct[0][n_demos-1] + 1
                    if len(nlp_relation) == 1 and nlp_relation[0]-11 == disp[0]:
                        nlp_correct[2][n_demos-1] = nlp_correct[2][n_demos-1] + 1
                if len(nlp_relation)== 1 and nlp_relation[0]-11 == disp[0]:
                    nlp_correct[1][n_demos-1] = nlp_correct[1][n_demos-1] + 1
                if lfd_shape_color[0] == landmark[3] and lfd_shape_color[1] == landmark[4]:
                    lfd_correct[0][n_demos-1] = lfd_correct[0][n_demos-1] + 1
                    if lfd_relation == disp[0]:
                        lfd_correct[2][n_demos-1] = lfd_correct[2][n_demos-1] + 1
                if lfd_relation == disp[0]:
                    lfd_correct[1][n_demos-1] = lfd_correct[1][n_demos-1] + 1
                if nlp_lfd_shape_color[0] == landmark[3] and nlp_lfd_shape_color[1] == landmark[4]:
                    nlp_lfd_correct[0][n_demos-1] = nlp_lfd_correct[0][n_demos-1] + 1
                    if nlp_lfd_relation == disp[0]:
                        nlp_lfd_correct[2][n_demos-1] = nlp_lfd_correct[2][n_demos-1] + 1
                if nlp_lfd_relation == disp[0]:
                    nlp_lfd_correct[1][n_demos-1] = nlp_lfd_correct[1][n_demos-1] + 1


            #guess placment randomly
            rand_placement = lfd.getRandomPointOnTable(user_id, exp_id, n_demos)
            #guess placement as average of demonstrated placements
            ave_placement = lfd.getMeanPlacementCoordinates(user_id, exp_id, n_demos)
      
            
            #compute accuracy over a test demo specified by demo_id
            for demo_id in range(max_demos, total_demos):
                #pure lfd error
                lfd_errors[i, n_demos-1] = nlp_lfd.averageDistanceError(lfd_shape_color, lfd_displacement, user_id, exp_id, max_demos, total_demos)
                #nlp+lfd error
                nlp_errors[i, n_demos-1] = nlp_lfd.averageDistanceError(nlp_lfd_shape_color, nlp_displacement, user_id, exp_id, max_demos, total_demos)
                #random baseline error
                random_errors[i, n_demos-1] = nlp_lfd.averageDistanceToPointError(rand_placement, user_id, exp_id, max_demos, total_demos)
                #average placement pos baseline error
                aveplace_errors[i, n_demos-1] = nlp_lfd.averageDistanceToPointError(ave_placement, user_id, exp_id, max_demos, total_demos)


    return (lfd_correct,nlp_correct, nlp_lfd_correct)
    
def plot(lfd_errors,nlp_errors,random_errors,aveplace_errors):
    max_demos = 4
    print ("lfd")
    print (lfd_errors)
    print ("nlp")
    print (nlp_errors)
    plt.figure()
    plt.errorbar(range(1,max_demos+1),np.mean(lfd_errors,0), yerr=np.std(lfd_errors,0), fmt='bo-', label='lfd')
    plt.errorbar(range(1,max_demos+1),np.mean(nlp_errors,0), yerr=np.std(nlp_errors,0), fmt='g^--', label='nlp+lfd')
    plt.errorbar(range(1,max_demos+1),np.mean(random_errors,0), yerr=np.std(random_errors,0), fmt='r*-.', label='random')
    plt.errorbar(range(1,max_demos+1),np.mean(aveplace_errors,0), yerr=np.std(aveplace_errors,0), fmt='ks', linestyle='dotted', label='ave. placement')
    plt.xticks(range(1,max_demos+1))
    plt.xlabel('number of demonstrations')
    plt.ylabel('generalization L2 error')
    plt.legend(loc='best')
    plt.show()
    


def main():
    warnings.filterwarnings("ignore")
    #train on first 25 users over all experiments and test on last 5 over all experiments
    #train_data = [(u,e) for u in range(25) for e in range(5)]
    #test_data = [(u,e) for u in range(25,30) for e in range(5)]
    data = []
    sub_user = []
    for u in range(30):
        for e in range(5):
            if(u%2 == 1 or e != 0):
                sub_user.append((u,e))
        if(u%2 == 1):
            data.append(sub_user)
            sub_user = []

    
    

    ######################
    #training code here
    ######################
    
    num_folds = 5#len(data)
   #kf = KFold(n=len(data),n_folds=num_folds)
    kFold(data,num_folds)



if __name__=="__main__":
    main()
