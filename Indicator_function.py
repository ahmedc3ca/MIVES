from math import *
import numpy as np
import matplotlib.pyplot as plt



def evaluate_function(P, inflection_point_coord, x_min,x_max, x,descending):
    C = inflection_point_coord[0] #Abscissa of infl. point
    K = inflection_point_coord[1] #Ordinate of infl. point
    if C == 0:
        C = 1e-3
    if K == 0:
        K = 1e-3
    exp_B = np.exp(-K*pow((abs(x_max-x_min)/C),P))
    B = pow(1-exp_B,-1)
    exp = np.exp(- K*pow((abs(x-x_min)/C),P))
    if descending:
        function = 1-B*(1-exp)
    else:
        function = B*(1-exp)
    return function

def plot_binary_function(descending):
    plt.clf()
    if descending:
        X = ["Yes","No"]
        Y = [1,0]
    else:
        X = ["Yes","No"]
        Y = [0,1]
    plt.bar(X,Y,width=0.1)
    plt.savefig("Plot.png",bbox_inches = "tight")


def plot_function(P, inflection_point_coord, x_min,x_max,descending):
    plt.clf()
    X = np.arange(x_min,x_max,(x_max-x_min)/1000)
    Y = []
    for x in X:
        Y.append(evaluate_function(P, inflection_point_coord, x_min,x_max, x,descending))
    plt.plot(X,Y)
    #plt.show()
    plt.savefig("Plot.png",bbox_inches = "tight")

def plot_linear_function(x_min,x_max,descending):
    P = 1
    inflection_point_coord = [1,0]
    plot_function(P,inflection_point_coord,x_min,x_max, descending)
    return P,inflection_point_coord
"""
# Exemple: linear function
P = 1
inflection_point_coord = [1,0]
x_min = 1
x_max = 5
plot_function(P, inflection_point_coord, x_min,x_max)

# GRAPHICS FOR THE SUBGRAPHICS TO BE SEEN
# LINEAR ONE:
P = 1
inflection_point_coord = [1,0]
x_min = 1
x_max = 5
plt.clf()
X = np.arange(x_min,x_max,1/1000)
Y = []
for x in X:
    Y.append(evaluate_function(P, inflection_point_coord, x_min,x_max, x,descending))
plt.plot(X,Y)
plt.savefig("LinearPlot.png",bbox_inches = "tight")



P = 3
inflection_point_coord = [3,0]
x_min = 1
x_max = 5
plt.clf()
X = np.arange(x_min,x_max,1/1000)
Y = []
for x in X:
    Y.append(evaluate_function(P, inflection_point_coord, x_min,x_max, x,descending))
plt.plot(X,Y)
plt.savefig("ConvexPlot.png",bbox_inches = "tight")
"""
# CONVEX ONE:
def plot_concave_function(x_min,x_max,descending):
    P = 3
    inflection_point_coord = [-1,0]
    plot_function(P,inflection_point_coord,x_min,x_max,descending)
    return P,inflection_point_coord


# CONCAVE ONE:
"""
P = 0.5
inflection_point_coord = [0.1,0.5]
x_min = 1
x_max = 5
plt.clf()
X = np.arange(x_min,x_max,1/1000)
Y = []
for x in X:
    Y.append(evaluate_function(P, inflection_point_coord, x_min,x_max, x,descending))
plt.plot(X,Y)
plt.savefig("ConcavePlot.png",bbox_inches = "tight")
#plt.show()
"""
def plot_convex_function(x_min,x_max,descending):
    P = 0.5
    inflection_point_coord = [0.1,0.5]
    plot_function(P,inflection_point_coord,x_min,x_max,descending)
    return P,inflection_point_coord

