# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:05:38 2018

@author: Cullen
"""

import pandas as pd
import numpy as np
import random
import shelve

from LSTM import LSTM

class neurnetAI(object):
    def __init__(self, magClasses, dirClasses):
        self.lRate = .001
        print("Loading magBrain")
        self.magBrain = LSTM('magBrain', 1, 'mag')
        print("DONE\n\nLoading dirBrain")
        self.dirBrain = LSTM('dirBrain', 1, 'dir')
        
        self.mags = np.zeros(magClasses)
        self.dirs = np.zeros(dirClasses)
        
        self.magT = 1
        self.dirT = 1
        
        self.magCounter = 0 #Counts how many training steps magBrain has been through
        self.dirCounter = 0 #Counts how many training steps dirbrain has been through
        
        
    #For learning as it plays
    def learning_step(self, packet, target):        
        canContinue, self.mags = self.train_limiter(self.mags, target[0])
        if canContinue:
            AImag = self.magBrain.learning_step(packet, target[0], self.lRate, self.magT)
            self.magCounter += 1
            if self.magCounter > 5000:
                self.magT += 1
                self.magCounter = 0
        else:
            AImag = 'limited'
        
        modifier = 360/self.dirs.shape[0]
        canContinue, self.dirs = self.train_limiter(self.dirs, int(target[1]/modifier))
        if canContinue:
            AIdir = self.dirBrain.learning_step(packet, target[1], self.lRate, self.dirT)
            self.dirCounter += 1
            if self.dirCounter > 5000:
                self.dirT += 1
                self.dirCounter = 0
        else:
            AIdir = 'limited'
            
        print('\nMag seen:', self.mags)
        print('Dir seen:', self.dirs)
        
        print("Mag:", self.magCounter, self.magT)
        print("Dir:", self.dirCounter, self.dirT)
        print('\n{}'.format([AImag, AIdir]))
        return [AImag, AIdir]
    
    
    #Limits the number of each variable seen to keep everything even
    def train_limiter(self, seen, target):
        if seen[target] - seen.min() >= 5:
            return False, seen
        else:
            seen[target] += 1
            return True, seen
    
    def save_model(self):
        self.magBrain.save_model()
        self.dirBrain.save_model()
        
    def get_move(self, packet):
        AIMag = round(self.magBrain.nnet_move(packet)[0])
        AIDir = round(self.dirBrain.nnet_move(packet)[0])
        
        return [AIMag, AIDir]
        
        
    def trainModel(self, fileName):
        def shuffleData(data):
            trainData = []
            while len(data) > 0:
                index = random.randint(0, len(data)-1)
                trainData.append(data[index])
                data.remove(data[index])
            return trainData
        
        try:
            db = shelve.open("{0}/{0}DB".format(fileName), "r")
            trainData = db['train']
            verData = db['ver']
            db.close()
        except:
            print("ERROR LOADING DATA")
         
        t = 1
        print("\nData loaded found:")
        print("{} training points".format(len(trainData)))
        print("{} verification points".format(len(verData)))
        print("Training model...")
        trainingResults = []
        columns = ["Epoch", "Magnitude Error", "Direction Error", "Training Error", " ",
                   "Verification Magnitude Error", "Verification Direction Error", "Verification Error"]
        
        for epoch in range(1,51):
            try:
                trainData = shuffleData(trainData)
                magError = 0
                dirError = 0
                verMagError = 0
                verDirError = 0
                for i in range(len(trainData)):
                    magError += self.magBrain.learning_step(trainData[i][0], trainData[i][-1][0], self.lRate, t, returnError = True)
                    dirError += self.dirBrain.learning_step(trainData[i][0], trainData[i][-1][1], self.lRate, t, returnError = True)
                
                magError /= i
                dirError /= i
                trainError = magError + dirError
                
                for i in range(len(verData)):
                    verMagError += self.magBrain.get_error(verData[i][0], verData[i][-1][0])
                    verDirError += self.dirBrain.get_error(verData[i][0], verData[i][-1][1])
                    
                verMagError /= i
                verDirError /= i
                verError = verMagError + verDirError
                
                print("\n=========EPOCH {}=========".format(epoch))
                print("magError:", magError)
                print("dirError:", dirError)
                print("trainError:", trainError)
                print("verMagError:", verMagError)
                print("verDirError:", verDirError)
                print("verError:", verError)
                container = [epoch, magError, dirError,trainError,verMagError,verDirError,verError]
                for i in range(len(container)):
                    container[i] = round(container[i], 3)
                container.insert(4, " ")
                trainingResults.append(container)
                
                
                t += 1
                if verError < .22:
                    break
            except KeyboardInterrupt:
                break
            
        df = pd.DataFrame(data = trainingResults)
        df.to_csv("training.csv", index = False, header = columns)
        self.save_model()
            
if __name__ == "__main__":
    nnetAI = neurnetAI(1,1)
    nnetAI.trainModel('good_data')
        
        
        