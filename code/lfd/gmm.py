import yaml
import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
import lfd


def learnBestMixtureModel(user_exp_tuples, num_demos):
    X = lfd.getAllDisplacementsConcat(user_exp_tuples, num_demos)
    #X = X[0:n_samples,:]
    #print X
    #print X.shape
    #print type(X)
    #print X.dtype


    lowest_bic = np.infty
    bic = []
    n_components_range = range(9, 10)
    cv_types = ['full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    #print bic
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange','red','green','pink','grey','purple'])
    print ("***learned", len(best_gmm.means_), "components")

    return best_gmm
#    bars = []
#    #print "fit models ............"
#    # Plot the BIC scores
#    spl = plt.subplot(2, 1, 1)
#    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#        xpos = np.array(n_components_range) + .2 * (i - 2)
#        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#                                      (i + 1) * len(n_components_range)],
#                            width=.2, color=color))
#    plt.xticks(n_components_range)
#    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
#    plt.title('BIC score per model')
#    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#        .2 * np.floor(bic.argmin() / len(n_components_range))
#    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
#    spl.set_xlabel('Number of components')
#    spl.legend([b[0] for b in bars], cv_types)

#    # Plot the winner
#    splot = plt.subplot(2, 1, 2)
#    Y_ = clf.predict(X)
#    print Y_
#    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                               color_iter)):
#        print "mean", mean
#        print "cov", cov    
#        print "cov shape", cov.shape
#        v, w = linalg.eigh(cov)
#        if not np.any(Y_ == i):
#            continue
#        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

#        # Plot an ellipse to show the Gaussian component
#        angle = np.arctan2(w[0][1], w[0][0])
#        angle = 180. * angle / np.pi  # convert to degrees
#        v = 2. * np.sqrt(2.) * np.sqrt(v)
#        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#        ell.set_clip_box(splot.bbox)
#        ell.set_alpha(.5)
#        splot.add_artist(ell)

#    plt.xticks(())
#    plt.yticks(())
#    plt.title('Selected GMM: full model, 2 components')
#    plt.subplots_adjust(hspace=.35, bottom=.02)
#    plt.show()


"""create dictionary for mapping from mixture ids to means and covariances"""
def writeModelToYaml(gm_model, filename):
    mu = {}
    cov = {}     
    for i in range(len(gm_model.means_)):
        mu[i] = gm_model.means_[i]
        cov[i] = gm_model.covariances_[i]
    data = {}
    data['mu'] = mu
    data['cov'] = cov
    stream = open(filename, 'w')
    yaml.dump(data, stream)

"""write out yaml files for each of the experiments in the training phase"""
def labelTrainingDataDisplacements(user_exp_tuples, num_demos):
    #learn model 
    gm_model = learnBestMixtureModel(user_exp_tuples, num_demos)

    #write model to file
    filename = "../data/gmm_params.yaml"
    writeModelToYaml(gm_model, filename)
    #print "learning gmm components"
    #write displacement yaml files 
    for user_id, exp_id in user_exp_tuples:
        #print "user", user_id
        for demo_id in range(num_demos):
            #find appropriate yaml file 
            yaml_file = lfd.getYamlFile(user_id, exp_id, demo_id)
            #grab displacements
            disp_list = lfd.getDisplacementList(yaml_file, user_id, exp_id)
            mixture_labels = []
            labels = gm_model.predict(np.array(disp_list))
            #label displacements
            for label in labels:
                mixture_labels.append([int(label)])
            #print "mixture labels", mixture_labels
            filename = "../data/user_"+str(user_id)+"/experiment_"+ str(exp_id)+"/displacements" + str(demo_id) + ".yaml"
            #print filename
            stream = open(filename, 'w')
            data = {}
            data['objects'] = mixture_labels
            #print data
            yaml.dump(data, stream)
    return gm_model
    
"""write out yaml files for each of the experiments in the training phase"""
def labelTestingDataDisplacements(user_exp_tuples, num_demos, model):
    #learn model 
    gm_model = model#learnBestMixtureModel(user_exp_tuples, num_demos)

    #write model to file
    filename = "../data/gmm_params.yaml"
    writeModelToYaml(gm_model, filename)
    #print "learning gmm components"
    #write displacement yaml files 
    for user_id, exp_id in user_exp_tuples:
        #print "user", user_id
        for demo_id in range(num_demos):
            #find appropriate yaml file 
            yaml_file = lfd.getYamlFile(user_id, exp_id, demo_id)
            #grab displacements
            disp_list = lfd.getDisplacementList(yaml_file, user_id, exp_id)
            mixture_labels = []
            labels = gm_model.predict(np.array(disp_list))
            #label displacements
            for label in labels:
                mixture_labels.append([int(label)])
            #print "mixture labels", mixture_labels
            filename = "../data/user_"+str(user_id)+"/experiment_"+ str(exp_id)+"/displacements" + str(demo_id) + ".yaml"
            #print filename
            stream = open(filename, 'w')
            data = {}
            data['objects'] = mixture_labels
            #print data
            yaml.dump(data, stream)

    

def main():
    #first learn mixtures and write mu and cov dicts to yaml file 
    user_exp_tuples = [(u,e) for u in range(30) for e in range(4)]
    num_demos = 10
#    gm_model = learnBestMixtureModel(user_exp_tuples, num_demos)
#    print "mean", gm_model.means_
#    print "cov", gm_model.covariances_
#    filename = "../data/gmm_params.yaml"
#    writeModelToYaml(gm_model, filename)
#    #how to access model data
#    model_data = lfd.getYamlData(filename)
#    print model_data['mu']
#    print model_data['cov']
    exp_id = 0
    user_id = 0
    demo_id = 0
    filename = "../data/user_"+str(user_id)+"/experiment_"+ str(exp_id)+"/displacements" + str(demo_id) + ".yaml"
    stream = open(filename, 'w')
    data = {}
    mixture_labels = [[0],[1],[2]]
    print( mixture_labels)
    data['objects'] = mixture_labels 
    print (data)
    yaml.dump(data, stream)
    labelTrainingDataDisplacements(user_exp_tuples, num_demos)



if __name__=="__main__":
    main()
