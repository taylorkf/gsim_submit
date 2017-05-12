from DMP.src.dmp import *
from linear_gaussian_controller.src.lgc import *
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == '__main__':
    f = sys.argv[1]
    p = sys.argv[2]
    x = float(sys.argv[3])
    y = float(sys.argv[4])
    gx = float(sys.argv[5])
    gy = float(sys.argv[6])

    demos = np.load(f)
    x0 = np.array([x, y])
    g = np.array([gx, gy])

    if p == 'dmp':
        demos = np.array([(np.array(range(len(d))), d) for d in demos])
 
        # train dmp
        K = 1500.0
        D = 2*np.sqrt(K)
        eps = 0.01
        spacing = uniformTime(eps)
        fapprox = radialBasisFunctApprox(15, spacing=spacing)
        dmp = DMP(K, D, eps, fapprox)
        dmp.learn(demos)
        v0 = np.array([0.0, 0.0])
        tau = 100.0
        dt = 1
        (t, x) = dmp.plan(x0, v0, g, tau, dt)
    else:
        gripper = np.array([d[:-1,:] for d in demos])
        goal = np.array([d[-1,:] for d in demos])

        state = np.array([np.concatenate((gripper[i][:-1], np.repeat([goal[i]], len(gripper[i][:-1]), axis=0), np.linalg.norm(gripper[i][:-1]-goal[i], axis=1)[:,np.newaxis]), axis=1) for i in range(len(demos))])
        action = np.array([np.array([gr[j+1]-gr[j] for j in range(len(gr)-1)]) for gr in gripper])

        # train lgc
        lgc = NthOrderLinearGaussianController(5)
        lgc.train(state, action)

        # plan path
        steps = 500
        x = [x0]
        for i in range(steps):
            state = np.concatenate((x[-1], g, [np.linalg.norm(x[-1]-g)]))
            action = lgc.step(state)
            x.append(x[-1] + action)
        x = np.array(x)

    # plot planned motion
    plt.plot(g[0], g[1], 'x')
    plt.plot(x[:,0], x[:,1])
    plt.xlim([0,1300])
    plt.ylim([0,1000])
    plt.show()

