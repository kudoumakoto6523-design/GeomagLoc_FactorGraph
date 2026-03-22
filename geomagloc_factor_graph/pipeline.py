'''
@ author: kudoumakoto
@ email: kudoumakoto6523@gmail.com
@ description:

This is the pipeline of the geomagnetic location algorithm using factor graph.
The details are as follows:
ATTENTION: This is not a algorithm based on SLAM or its relative algorithms. This is a prebuild map-based
algorithm, using geomagnetic fingerprint.
VERSION DECLARATION: For simplify the algorithm, the motion model will be pedestrian dead reckon (PDR).
The repo is acclaimed to be universal as the code improves in the promising future.

As I cannot draw a factor graph directly, a description can be given as follows:
Connection between two states is consisted by PDR(step length and heading angle), and the observation of geomagnetic
intensity is only related to the current state, which strictly obeys the assumptions in Hidden Markov Model (HMM)

'''

import simulation as sim


def pipeline(source="uji", data_root="data/raw"):
    n = 11
    W = 100
    map = sim.get_mag(source=source, data_root=data_root)
    for i in range(n):
        templist = []
        geomag_list = []
        pos_list = []
        route = sim.get_true_route()
        pos_list = [sim.initialize()]
        l = sim.get_test_len()
        for j in range(l):
            mag, acc, gyro = sim.get_sensor()
            templist.append([acc, gyro, mag])
            judge = sim.judge_step(templist)
            if judge:
                stplen = sim.get_step_len(templist)
                heading_angle = sim.get_heading_angle(templist)
                geomag = sim.get_mag()
                geomag_list.append(geomag)
                geomag_list_using = geomag_list[-W:]
                pos = sim.factor_graph(stplen, heading_angle, geomag_list_using)
                pos_list.append(pos)
                templist.clear()
            else:
                continue

