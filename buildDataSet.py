# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:06:11 2018

@author: Cullen
"""
import numpy as np
import random
import shelve
import os
from Engine import Engine

def shuffle(data):
    new = []
    while len(data) > 0:
        index = random.randint(0, len(data)-1)
        new.append(data[index])
        data.remove(data[index])
    return new

def mergeData(file1, file2):
    db = shelve.open("{0}/{0}DB".format(file1), "w")
    data1 = db['data']
    
    db2 = shelve.open("{0}/{0}DB".format(file2), "r")
    data2 = db2['data']   
    db2.close()
        
    data = data1 + data2
    
    db['data'] = data
    
    db.close()
    

def cleanData(fileName):
    db = shelve.open("{0}/{0}DB".format(fileName), "r")
    data = db['data']
        
    #Count the number of classes for the data
    numMagData = np.zeros(11)
    numDirData = np.zeros(36)
    for item in data:
        numMagData[item[-1][0]] += 1
        #In case it rounds 356.x to 36, which is out of index range
        try:
            numDirData[round(item[-1][1]/10)] += 1
        except:
            numDirData[0] += 1
    numMagTrain = round(np.min(numMagData) * .85)
    numDirTrain = round(np.min(numDirData) * .85)
    numMagVer = 40
    numDirVer = 12
    
    t = 0
    while True:
        data = shuffle(data)
        nonrandomTrainData = []
        verData = []
        numMagTrainArray = np.zeros(11)
        numDirTrainArray = np.zeros(36)
        numMagVerArray = np.zeros(11)
        numDirVerArray = np.zeros(36)
        zer = []
        for item in data:
            if item[-1][0] == 0:
                zer.append(item)
                continue
            try:
                dirClass = round(item[-1][1]/10)
            except:
                dirClass = 0
            if numMagTrainArray[item[-1][0]] < numMagTrain and numDirTrainArray[dirClass] < numDirTrain:
                nonrandomTrainData.append(item)
                numMagTrainArray[item[-1][0]] += 1
                numDirTrainArray[dirClass] += 1
            elif numMagVerArray[item[-1][0]] < numMagVer and numDirVerArray[dirClass] < numDirVer:
                verData.append(item)
                numMagVerArray[item[-1][0]] += 1
                numDirVerArray[dirClass] += 1
        for item in zer:
            try:
                dirClass = round(item[-1][1]/10)
            except:
                dirClass = 0
            if numMagTrainArray[item[-1][0]] < numMagTrain and numDirTrainArray[dirClass] < numDirTrain:
                nonrandomTrainData.append(item)
                numMagTrainArray[item[-1][0]] += 1
                numDirTrainArray[dirClass] += 1
            elif numMagVerArray[item[-1][0]] < numMagVer and numDirVerArray[dirClass] < numDirVer:
                verData.append(item)
                numMagVerArray[item[-1][0]] += 1
                numDirVerArray[dirClass] += 1
        
        print(numMagTrainArray)
        print(numDirTrainArray)
        print(numMagVerArray)
        print(numDirVerArray)
        
        if np.max(numMagTrainArray) - np.min(numMagTrainArray) < 5 and np.max(numDirTrainArray) - np.min(numDirTrainArray) < 5:
            break
        elif t > 30:
            break
        else:
            numMagTrain = round(np.average(numMagTrainArray))
            numDirTrain = round(np.average(numDirTrainArray))
            t += 1
            
    #Shuffle the training data
    trainData = shuffle(nonrandomTrainData)

    db['train'] = trainData
    db['ver'] = verData
    db.close()

    
    

def buildDataSet(fileName):
    data = []
    redSize = random.randint(1,2)
    blueSize = random.randint(1,2)
    engine = Engine(redSize, blueSize, 'minmax', 'minmax')
    numMag = [0] * 11
    numDir = [0] * 36
    magTarget = 720
    dirTarget = 220
    
    for i in range(10000):
        try:
            turn = 0
            while not engine.checkEndState() and turn < 5:
                turnData = engine.doOneTurn()
                if numMag[turnData[-1][0]] < magTarget or numDir[int(turnData[-1][1]/10)] < dirTarget:
                    numMag[turnData[-1][0]] += 1
                    numDir[int(turnData[-1][1]/10)] += 1
                    data.append(turnData)
                turn += 1
                
            print(numMag)
            print(numDir)
            
            redSize = random.randint(1,2)
            blueSize = random.randint(1,2)
            engine.reset(redSize, blueSize, 'minmax', 'minmax', allowRandom = True)
        except KeyboardInterrupt:
            try:
                db = shelve.open("{0}/{0}DB".format(fileName), "r")
                db.close()
            except:
                try:
                    os.makedirs('{0}'.format(fileName))
                except:
                    pass
                db = shelve.open("{0}/{0}DB".format(fileName), 'c')
                db.close()
            finally:
                db = shelve.open("{0}/{0}DB".format(fileName), "w")
                db['data'] = data
                db.close()
            break
    
if __name__ == "__main__":
    buildDataSet('good_data2')
#    cleanData('good_data')