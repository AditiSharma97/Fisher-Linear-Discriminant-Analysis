import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats

file = "dataset_1.csv"
data = pd.read_csv (file, header=None)
noOfTrainingExamples = 1000
x = data.iloc[0:noOfTrainingExamples, 1:3].values #(1000,2)
y = data.iloc[0:noOfTrainingExamples, 3].values
positiveCount = np.count_nonzero(y)
negativeCount = noOfTrainingExamples - positiveCount
positiveSum = np.zeros((1,2))
negativeSum = np.zeros((1,2))
for i in range (noOfTrainingExamples):
	if y[i] == 1:
		positiveSum += x[i]
	else:
		negativeSum += x[i]
print (positiveSum)	
print (negativeSum)
positiveExamplesMean = positiveSum/positiveCount
negativeExamplesMean = negativeSum/negativeCount
print (positiveExamplesMean)
print (negativeExamplesMean)
print (positiveSum.shape)
print (positiveExamplesMean.shape)

A = np.zeros((positiveCount,2))
B = np.zeros((negativeCount,2))
i = 0
j = 0
for k in range(noOfTrainingExamples):
	if y[k] == 1:
		A[i] = x[k] - positiveExamplesMean
		i = i + 1
	else:
		B[j] = x[k] - negativeExamplesMean
		j = j + 1
print (A.size)
#print (A) 
print ("B")
#print (B)
#Sw = A*A.T + B*B.T
Sw = np.add(np.dot(A.T, A), np.dot(B.T, B))
#print (Sw)
w = np.dot(np.linalg.inv(Sw), (negativeExamplesMean - positiveExamplesMean).T) #(2,1)
#print ("W")
#print (w)


projectedX = np.dot(w.T, x.T).T #(1000,1)
print ("project")
print (projectedX)
positivePoints = []
negativePoints = []
for i in range (noOfTrainingExamples):
	if y[i] == 1:
		positivePoints.append(projectedX[i])
	else:
		negativePoints.append(projectedX[i])
positivePoints.sort()
positivePointsMean = np.mean(positivePoints)
positivePointsStd = np.std(positivePoints)
negativePoints.sort()
negativePointsMean = np.mean (negativePoints)
negativePointsStd = np.std (negativePoints)
pdf1 = stats.norm.pdf (positivePoints, positivePointsMean, positivePointsStd)
pdf2 = stats.norm.pdf (negativePoints, negativePointsMean, negativePointsStd)
plt.plot(positivePoints, pdf1)
plt.plot(negativePoints, pdf2)

a = 1/(2*negativePointsStd**2) - 1/(2*positivePointsStd**2)
b = positivePointsMean/(positivePointsStd**2) - negativePointsMean/(negativePointsStd**2)
c = negativePointsMean**2 /(2*negativePointsStd**2) - positivePointsMean**2 / (2*positivePointsStd**2) - np.log(positivePointsStd/negativePointsStd)
result = np.roots([a,b,c])
result = solve(pointsY0mean,pointsY1mean,pointsY0std,pointsY1std)

plotNegativeX = []
plotNegativeY = []
for i in range(negativeCount):
    plotNegativeX.append(0)
    plotNegativeY.append(negativePoints[i])
plt.scatter(plotNegativeY, plotNegativeX, color = 'blue')

plotPositiveX = []
plotPositiveY = []
for i in range(positiveCount):
    plotPositiveX.append(0)
    plotPositiveY.append(positivePoints[i])
plt.scatter(plotPositiveY, plotPositiveX, color = 'red')

plt.show()
