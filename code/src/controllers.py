from linear_gaussian_controller.src.lgc import *
from DMP.src.dmp import *
import numpy as np

class Controller(object):
    def __init__(self):
        pass

    def train(self, demos):
        pass

    def playback(self, demos):
        pass

    def dist(self, a, b):
        return np.linalg.norm(a-b)

class DMPController(Controller):
    def __init__(self):
        self.K = 1500.0
        self.D = 2*np.sqrt(self.K)
        self.B = 15
        self.eps = 0.01
        self.tau = 100.0
        self.dt = 1
        self.t = 20
        self.spacing = uniformTime(self.eps)
        self.dmp = None

    def train(self, demos):
        demos = np.array([(np.array(range(len(d))), d) for d in demos])
        N = len(demos)
        
        # train dmp
        if N == 1:
            fapprox = linearInterpFunctApprox
        elif N >= 1:
            fapprox = radialBasisFunctApprox(self.B, spacing=self.spacing)
        else:
            raise Exception("No demonstrations provided")

        self.dmp = DMP(self.K, self.D, self.eps, fapprox)
        self.dmp.learn(demos)

    def playback(self, s, g):
        v0 = np.zeros(len(s))
        (_, x) = self.dmp.plan(s, v0, g, self.tau, self.dt)

        idxs = []
        for i in range(len(x)):
            if self.dist(g, x[i,:]) >= self.t:
                idxs.append(i)
            else:
                return x[idxs,:]

        return x


class LGCController(Controller):
    def __init__(self):
        self.order = 5
        self.steps = 500
        self.t = 20
        self.lgc = NthOrderLinearGaussianController(self.order)

    def train(self, demos):
        gripper = np.array([d[:-1,:] for d in demos])
        goal = np.array([d[-1,:] for d in demos])

        state = np.array([np.concatenate((gripper[i][:-1], np.repeat([goal[i]], len(gripper[i][:-1]), axis=0), np.linalg.norm(gripper[i][:-1]-goal[i], axis=1)[:,np.newaxis]), axis=1) for i in range(len(demos))])
        action = np.array([np.array([gr[j+1]-gr[j] for j in range(len(gr)-1)]) for gr in gripper])
        
        # train lgc
        self.lgc.train(state, action)

    def playback(self, s, g):
        x = [s]

        for i in range(self.steps):
            state = np.concatenate((x[-1], g, [np.linalg.norm(x[-1]-g)]))
            action = self.lgc.step(state)

            if self.dist(g, x[-1] + action) >= self.t:
                x.append(x[-1] + action)
            else:
                return np.array(x)

        return np.array(x)

