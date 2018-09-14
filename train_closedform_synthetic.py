from sklearn.cluster import KMeans
from pandas import read_csv
from numpy import identity,matrix,var,vstack,cov,matmul
from math import exp,sqrt
from numpy.linalg import pinv

def getSpreads(data,M,clusterIndexes):
    spreads = []
    temp=[]
    for i in range(0,M):
        temp=[]
        for j in range(1,len(clusterIndexes)+1):
            if clusterIndexes[j-1]==i:
                temp.append(data[j-1:j])
        ds=vstack((temp))
        spreads.append(cov(ds,rowvar=False))
    return spreads

def getDesignMatrix(data,M):    
    kmeans = KMeans(n_clusters=M, random_state=0).fit(data)
    centroids=kmeans.cluster_centers_
    spreads = getSpreads(data,M,kmeans.labels_)

    phi=[]
    for i in range(0,M):
        temp=[]
        for j in range(1,len(kmeans.labels_)+1):
            t=matrix(data[j-1:j]-centroids[i])
            temp.append(exp((-1*matmul(matmul(t,matrix(pinv(spreads[i]*identity(len(spreads[i]))))),t.transpose()).tolist()[0][0])/2))
        phi.append(temp)
    phi=matrix(phi).transpose()
    return phi,centroids,spreads

def getWeights(lambdaval,M,phi,trainingOutput):
    weight=matmul(matmul(pinv((lambdaval*identity(M))+matmul(phi.transpose(),phi)),phi.transpose()),matrix(trainingOutput))
    return weight
b=0
def getRootMeanSquareError(M,validationOutput,phi,lambdaval,w,size):
    error=0
    b=w
    print(w)
    for i in range(0,len(validationOutput)):
        t=0
        for j in range(1,M):
            t+=w[j]*phi.item(i,j)
        print(i)
        error+= (validationOutput[0][i+1600]-t)**2
        error+=lambdaval*matmul(w.transpose(),w)
        rms=sqrt(error/len(validationOutput))
    return rms

def trainClosedForm(data,outputLabels):
    trainingData=data[0:55000]
    validationData=data[55000:62000]
    testData=data[62000:69000]

    trainingOutput=outputLabels[0:55000]
    validationOutput=outputLabels[55000:62600]
    testOutput=outputLabels[62000:69000]
    size=len(data)

    lambdaval=0.01
    minimalPoint=0
    learningrate=0
    preverror=float("inf")
    prevTrainingError=float("inf")
    for M in range(2,3):
        phi,mu,spread=getDesignMatrix(trainingData,M)
        phiValidate,muval,spreadval=getDesignMatrix(validationData,M)
        while(lambdaval<10):
            w= getWeights(lambdaval,M,phi,trainingOutput)
            error=0
            b=w
            for i in range(0,len(validationOutput)):
                t=0
                for j in range(1,M):
                    t+=w[j][0]*phiValidate.item(i,j)
                error+= (validationOutput[0][i+int(55000)]-t)**2
                error+=lambdaval*matmul(w.transpose(),w)
            validationError=sqrt(error/len(validationOutput))
            if(preverror>validationError):
                weights=w
                preverror=validationError
                finalMu=mu
                finalSpread=spread
                finalLambda=lambdaval
                clusterSize=M
            else:
                    minimalPoint=1
                    break
            lambdaval*=2
        if minimalPoint==1:
            break
    return weights,finalLambda,clusterSize,finalMu,finalSpread,learningrate