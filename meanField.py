import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import mnist
import scipy.misc
import copy as copy


# Function to display image
# https://stackoverflow.com/questions/42353676/display-mnist-image-using-matplotlib
def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    plt.show()

# Function to replace pixels in certain images with noise
def noise_img(data, noiseData):

    # Iterate through each col of the noise data,
    # this is the value that will be used as the noise bit
    for col in range(noiseData.shape[1]):

        # Counters for iterating two rows at a time
        i = 0
        maxRow = noiseData.shape[0]

        # Iterate through the rows of the noise data
        while i < maxRow:
            # Store Index for pixel being replaced with noise
            rowIdx = noiseData[i, col]
            colIdx = noiseData[i+1, col]
            imageNum = i/2

            # print("noise col " + str(col) + " row " + str(i))
            # print("rowIdx " + str(rowIdx) + " colIdx " + str(colIdx) + " imgNum " + str(imageNum))
            # Replace pixel in imageNum rowIdx colIdx with the col index
            data[int(imageNum), int(rowIdx), int(colIdx)] = col

            # Skip a row
            i = i + 2


# Returns list with the neighbors from provided coords
def neighbors(xCoord, yCoord):

    neighbors = []
    tmpx = 0
    tmpy = 0

    # If statements to ensure we are not in a edge

    # Start W/ west coord first
    if xCoord != 0:
        tmpx = xCoord -1
        tmpy = yCoord
        neighbors.append([tmpx, tmpy])

    # North Coord Next
    if yCoord != 0:
        tmpx = xCoord
        tmpy = yCoord -1
        neighbors.append([tmpx, tmpy])

    # East Coord
    if xCoord < 28:
        tmpx = xCoord + 1
        tmpy = yCoord
        neighbors.append([tmpx, tmpy])

    # South Coord
    if xCoord < 28:
        tmpx = xCoord
        tmpy = yCoord + 1
        neighbors.append([tmpx, tmpy])

    return neighbors



# # PART 1
# Read MNIST train images using the mnist library
images = mnist.train_images()
train_labels = mnist.train_labels()

# Get first 20 images
X = images[0:20]
Y = train_labels[0:20]
gen_image(X[0])

# Binarize images
# Divide by 255, then round using rint. Values < 0.5 = 0  and  >= 0.5 = 1
# We then swap all values of 0 for -1
binX = X/255
binX = np.rint(binX)
binX[binX == 0] = -1
gen_image(binX[0])


# PART 2
# Read Noise file & call function to replace bits w/ noise
noiseX = binX
noise = np.genfromtxt('SupplementaryAndSampleData/NoiseCoordinates.csv', delimiter=',',
                      skip_header=1, usecols=range(1, 16))
noise_img(noiseX, noise)
gen_image(noiseX[0])


# PART 3
# Build some Bolts and Machines, I mean a Boltzman Machine
thetaH = 0.8
thetaX = 2.0
iterations = 10
E = 10e-10

Q = np.genfromtxt('SupplementaryAndSampleData/InitialParametersModel.csv', delimiter=',')
updateOrder = np.genfromtxt('SupplementaryAndSampleData/UpdateOrderCoordinates.csv', delimiter=',',
                      skip_header=1, usecols=range(1, 785))


# General Steps To Perform
# https://piazza.com/class/jchzguhsowz6n9?cid=1295
# 1. load initial parms for q (Q)
# 2. Calculate Pi using Qrc[H=1]*** Follow order provided
# 3. Performs 10 Iters
# 4. Traverse through Pi, if P_ij > 0.5 then 1 else 0. This is the new image
#

# Start with 10 Images for testing, to match the provided sample
X = noiseX[0:10]
updateOrderTmp = updateOrder[0:20]

# Copy Q for each image, so all images start with the same Q values
q = np.zeros((10,28,28))
for i in range(10):
    q[i] = Q


# Calculate the Pi values

# EQ = SUM( Qrc[Hrc = 1] log( Qrc[Hr,c = 1] + E) + Qr,c[Hrc = 1] log(Qrc[Hrc = -1] + E  )

# https://piazza.com/class/jchzguhsowz6n9?cid=1341
# def update_Q(Q,i,j):
#    above = (np.e)**(sum(phetaHH*EqH(Q,i_,j_) for i_,j_ in neighbor(i,j))+phetaHX*X[i,j])
#     below = above+(np.e)**(-np.log(above))
#     Q[i,j] = above/below

i = 26
j = 4
Qi = 0

t1 = 0

qeh = copy.deepcopy(Q)

# First term of numerator
#  SumOverNeighbors ThetaH * (2Pi_ij - 1)
for i_, j_ in neighbors(i, j):
    print(str(i_) + " " + str(j_))
    t1 = t1 + (thetaH *((2 * qeh[i_, j_]) - 1))

# Second term of numerator
# Summation of Neighbors in X  ThethaX * Xj
t2 = thetaX*X[0, i, j]

# Numerator e ^ (term1 + term2)
Numerator = np.e ** (t1 + t2)

# Denominator T1 = Numerator
# Denominator T2 part 1
# SumOverNeighbors -ThetaH * (2Pi_ij - 1)
t1 = 0
for i_, j_ in neighbors(i, j):
    print(str(i_) + " " + str(j_))
    t1 = t1 + ((-1*thetaH) *((2 * qeh[i_, j_]) - 1))

# Denominator T2 part 2
# Summation of Neighbors in X  -ThethaX * Xj
t2 = (-1*thetaX)*X[0, i, j]

#Denominator = numerator + e^(t1 + t2)
denominator = Numerator + np.e ** (t1 + t2)

updatedPi = Numerator / denominator