import numpy as np
from matplotlib import pyplot as plt
import mnist
import copy as copy


# Function to display image
# https://stackoverflow.com/questions/42353676/display-mnist-image-using-matplotlib
def gen_image(arr):
    two_d = (np.reshape(arr, (arr.shape[0], arr.shape[1])) * 255).astype(np.uint8)
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
            data[int(imageNum), int(rowIdx), int(colIdx)] *= -1

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
    if xCoord < 27:
        tmpx = xCoord + 1
        tmpy = yCoord
        neighbors.append([tmpx, tmpy])

    # South Coord
    if yCoord < 27:
        tmpx = xCoord
        tmpy = yCoord + 1
        neighbors.append([tmpx, tmpy])

    return neighbors


# calculates term1 of the Eq[logP[H,X]] calculation
# Summation in Neighbors of ij in H of  Thetha * 2qk[Hk = 1] − 1
def calc_logP_term1(Qk, i, j, thetaH):

    p1 = 0
    # Iterate through neighbors
    for i_, j_ in neighbors(i, j):
        p1 += thetaH * (2 * Qk[i, j] - 1) * (2 * Qk[i_, j_] - 1)

    return p1



# calculates term2 of the Eq[logP[H,X]] calculation
# Summation in Neighbors of ij in X  Theta * Eqi[Hi]Xj
def calc_logP_term2(Qk, i, j, thetaX, Xi):

    # Calc second term
    p2 = thetaX * (2 * Qk[i, j] - 1) * Xi[i, j]
    return p2


# Function to update EQ
def update_EQ(H, thetaH, thetaX, x):

    E = 10e-10  # Tiny Error
    p = np.zeros((H.shape[0], 1))
    q = np.zeros((H.shape[0], 1))

    # Iterate through each image in hidden matrix
    for img in range(H.shape[0]):

        # Iterate through rows & cols
        for row in range(H.shape[1]):
            for col in range(H.shape[2]):

                # Eq[Log[H,X]
                p[img] += calc_logP_term1(H[img], row, col, thetaH) + calc_logP_term2(H[img], row, col, thetaX, x[img])

                # Eq[logQ]
                # Q[H=1] * Log(Q[H=1] + E) + Q[H=-1] * log(Q[H=-1] + E
                q[img] += H[img, row, col] * np.log((H[img, row, col] + E)) + \
                     (1 - H[img, row, col]) * np.log(((1 - H[img, row, col]) + E))

    return q - p


# # PART 1
# Read MNIST train images using the mnist library
images = mnist.train_images()

# Get first 20 images
X = images[0:20]
gen_image(X[0])

# Binarize images
# Divide by 255, then round using rint. Values < 0.5 = 0  and  >= 0.5 = 1
# We then swap all values of 0 for -1
binX = copy.deepcopy(X)
binX = binX/255
binX = np.rint(binX)
binX[binX == 0] = -1
gen_image(binX[0])


# PART 2
# Read Noise file & call function to replace bits w/ noise
noiseX = copy.deepcopy(binX)
noise = np.genfromtxt('SupplementaryAndSampleData/NoiseCoordinates.csv', delimiter=',',
                      skip_header=1, usecols=range(1, 16))

# Call noise function to add noise bits
noise_img(noiseX, noise)
gen_image(noiseX[0])


# PART 3
# Build some Bolts and Machines, I mean a Boltzman Machine
thetaH = 0.8
thetaX = 2.0
iterations = 10

# Initial params
Q = np.genfromtxt('SupplementaryAndSampleData/InitialParametersModel.csv', delimiter=',')

# Update order
updateOrder = np.genfromtxt('SupplementaryAndSampleData/UpdateOrderCoordinates.csv', delimiter=',',
                      skip_header=1, usecols=range(1, 785))


# Copy Q for each image, so all images start with the same Q values
EQ = np.zeros((20, 28, 28))
for i in range(20):
    EQ[i] = Q


# Adjust for 10 images
EQ = EQ[0:10]
noiseX = noiseX[10:20]
updateOrder = updateOrder[20:40, :]

# List to store the energy after each iteration
energyList = []
for iters in range(iterations):
    print("Iter " + str(iters))

    # Store Energy for each iteration
    energyList.append(update_EQ(EQ, thetaH, thetaX, noiseX))

    # Update Pi in order
    # Iterate through rows, skipping the second
    for i in range(0, updateOrder.shape[0], 2):
        for j in range(updateOrder.shape[1]):

            # Index and image number
            rowIdx = int(updateOrder[i, j])
            colIdx = int(updateOrder[i+1, j])
            imageNum = int(i/2)

            # Update Pi
            t1 = 0
            t2 = 0

            # Get neighbors and iterate
            for i_, j_ in neighbors(rowIdx, colIdx):
                #  ThetaH * (2Pi_ij - 1)  -  First term in the numerator
                t1 += thetaH * ((2 * EQ[imageNum, i_, j_]) - 1)

                # -ThetaH * (2Pi_ij - 1)  - First part of the second term in denominator
                t2 += ((-1 * thetaH) * ((2 * EQ[imageNum, i_, j_]) - 1))

            # Numerator e ^ (term1 + thetaX * X)
            numerator = np.e ** (t1 + (thetaX * noiseX[imageNum, rowIdx, colIdx]))

            # Denominator = numerator + e^(t1 + (-thetaX * X )
            denominator = numerator + np.e ** (t2 + (-1*thetaX) * noiseX[imageNum, rowIdx, colIdx] )

            # Update Pi
            EQ[imageNum, rowIdx, colIdx] = numerator / denominator


# Transform energy and write to csv
energy = np.array(energyList).reshape(10, energyList[0].shape[0]).T
np.savetxt("energyResult.csv", energy, delimiter=',')

# Array to hold reconstructed images by col
hidden_img = np.zeros((EQ.shape[1], EQ.shape[0] * EQ.shape[2]))

# Reshape Images into cols
for img in range(EQ.shape[0]):
    startInd = img * 28
    endInd = startInd + 28
    hidden_img[:, startInd:endInd] = EQ[img]

# Display Hidden Images
gen_image(hidden_img)

# MAP Images by converting hidden pixels 0 if EQij < 0.5 or 1 if EQij >= 0.5
map_img = copy.deepcopy(hidden_img)
for i in range(map_img.shape[0]):
    for j in range(map_img.shape[1]):
            if map_img[i][j] >= 0.5:
                map_img[i][j] = 1
            else:
                map_img[i][j] = 0

# Display MAP Images and save into csv
gen_image(map_img)
np.savetxt("denoised.csv", map_img, fmt='%d', delimiter=',')


# Save energy.csv for only first two images for the first iteration
# For autograder submission
eng = energy[0:2, 0:2]
np.savetxt("energy.csv", eng, delimiter=',')

















