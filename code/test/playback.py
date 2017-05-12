from gsim.src.sim import *
import numpy as np
import sys

if __name__ == '__main__':
    w_path = sys.argv[1]
    m_path = sys.argv[2]
    n = int(sys.argv[3])
    p = sys.argv[4]
    s = np.array(map(float, sys.argv[5][1:-1].split(',')))
    g = np.array(map(float, sys.argv[6][1:-1].split(',')))

    if p == 'dmp':
        policy = DMPController()
    else:
        policy = LGCController()

    demos = np.load(m_path)[:n]
    policy.train(demos)

    playback = GSimPlayback(1300, 1000, w_path, policy)
    playback.run(s, g)

