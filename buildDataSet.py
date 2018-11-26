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
from maxnetAI import maxnetAI

def minimizePacket(packet):
    for i in range(len(packet)):
        packet[i].pop(2) #Remove fireRange
        packet[i].pop(2) #Remove moveSpeed
        packet[i].pop(2) #Remove x-coord
        packet[i].pop(2) #Remove y-coord
        packet[i][1] = int(round(packet[i][1])) #Round heading
        packet[i][2] = int(round(packet[i][2])) #Round heading
        packet[i][3] = int(round(packet[i][3])) #Round distance
        packet[i][4] = int(round(packet[i][4])) #Round direction
    return packet

def getMoveFromQTable(Q):
    action = np.argmax(Q)
    angleStepSize = 360//(len(Q)//11)
    magAndDirModifier = 360//angleStepSize
    magnitude = action // magAndDirModifier
    direction = (action % magAndDirModifier) * angleStepSize
    return [magnitude, direction]

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
    
#    data = []
    
#    for item in data2:
#        if item in data1:
#            continue
#        data.append(item)
        
    data = data1 + data2
    
    db['data'] = data
    
    db.close()
    
def relabelData(fileName):
    db = shelve.open("{0}/{0}DB".format(fileName), "r")
    data = db['data']
    db.close()
    print("relabeling {} data points.".format(len(data)))
    maxnet = maxnetAI()
    
    for i in range(len(data)):
        try:
            move = maxnet.get_move(data[i][0])
            data[i][-1] = move
            print(i, len(data))
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
                print("SAVING DATA")
                db = shelve.open("{0}/{0}DB".format(fileName), "w")
                db['data1'] = data
                db.close()
            break
    

def cleanData(fileName):
    db = shelve.open("{0}/{0}DB".format(fileName), "r")
    data = db['data']
    db.close()
        
    #Count the number of classes for the data
    numMagData = np.zeros(11)
    numDirData = np.zeros(72)
    newData = []
    for item in data:
        newData.append([item[0], getMoveFromQTable(item[1])])
    data = newData
    for item in data:
        numMagData[item[-1][0]] += 1
        #In case it rounds 356.x to 36, which is out of index range
        try:
            numDirData[int(round(item[-1][1]/5))] += 1
        except:
            numDirData[0] += 1
    numMagTrain = round(np.min(numMagData) * .7)
    numDirTrain = round(np.min(numDirData) * .7)
    numMagVer = 65
    numDirVer = 20
    
    t = 0
    while True:
        data = shuffle(data)
        nonrandomTrainData = []
        verData = []
        numMagTrainArray = np.zeros(11)
        numDirTrainArray = np.zeros(72)
        numMagVerArray = np.zeros(11)
        numDirVerArray = np.zeros(72)
        zer = []
        for item in data:
            if item[-1][0] == 0:
                zer.append(item)
                continue
            try:
                dirClass = int(round(item[-1][1]/5))
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
                dirClass = int(round(item[-1][1]/5))
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
        elif t > 100:
            break
        else:
            numMagTrain = np.floor(np.average(numMagTrainArray))
            numDirTrain = np.floor(np.average(numDirTrainArray))
            t += 1
            
    #Shuffle the training data
    trainData = shuffle(nonrandomTrainData)
    db = shelve.open("{0}/{0}DB".format(fileName), "w")
    db['train'] = trainData
    db['ver'] = verData
    db.close()

    
    

def buildDataSet(fileName):
    data = []
    sizeArray = [random.randint(1,2), random.randint(1,2)]
    controllers = ['maxnet', 'maxnet']
    engine = Engine(sizeArray, controllers)
    numMag = [0] * 11
    numDir = [0] * 36
    magTarget = 900
    dirTarget = 400
    
    for i in range(10000):
        try:
            turn = 0
            while not engine.checkEndState() and turn < 5:
                turnData = engine.doOneTurn()
                if numMag[turnData[-1][0]] < magTarget or numDir[int(turnData[-1][1]/10)] < dirTarget:
                    numMag[turnData[-1][0]] += 1
                    try:
                        numDir[int(round(turnData[-1][1]/10))] += 1
                    except:
                        numDir[0] += 1
                    data.append(turnData)
                turn += 1
                
            print(numMag)
            print(numDir)
            
            sizeArray = [random.randint(1,2), random.randint(1,2)]
            engine.reset(sizeArray, controllers, allowRandom = True)
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
                print("SAVING DATA")
                db = shelve.open("{0}/{0}DB".format(fileName), "w")
                db['data'] = data
                db.close()
            break
    
if __name__ == "__main__":
#    buildDataSet('maxnet_data2')
#    mergeData('maxnet_data', 'maxnet_data2')
#    relabelData('data')
    
    cleanData('sarsa')