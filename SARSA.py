# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:49:06 2018

@author: Cullen
"""

import numpy as np
import shelve
import random
import time
import os
from Utility import Utility
from collections import defaultdict
from copy import deepcopy
from AI import AI

utilities = Utility()

class SARSA(AI):
    def __init__(self, angleStepSize = 5):
        self.angleStepSize = angleStepSize
        self.magAndDirModifier = 360//angleStepSize
        self.actionSpaceSize = 11 * self.magAndDirModifier
        self.stateList = {}
        self.Q = defaultdict(lambda: np.zeros(self.actionSpaceSize))
        self.eps = 1.
        self.gamma = 1.
        self.alpha = .5
        
        self.Walls = []
        self.wallHash = -1
            
        
    def loadMemories(self, wallHash):
        try:
            path = 'sarsa/{0}/{0}DB'.format(wallHash)
            db = shelve.open(path, 'r')
            data = db['data']
            db.close()
            for i in range(len(data)):
                hashValue, _ = self.getIndexForState(data[i][0])
                self.stateList[hashValue][1] = data[i][2]
                self.Q[hashValue] = data[i][1]
        except:
            os.makedirs('sarsa/{0}'.format(wallHash))
        
    def getIndexForState(self, state, stateListToCheckAgainst = None):
        minimalPacket = self.minimizePacket(deepcopy(state)) #Don't want to override the packet so deepcopy
        #minimalPacket is empty (all armies dead)
        if len(minimalPacket) == 0:
            if not stateListToCheckAgainst:
                self.stateList[0] = [[], 1, []]
                return 0, 0
            else:
                stateListToCheckAgainst[0] = [[], 1, []]
                return 0, stateListToCheckAgainst
        else:
            hashValue = 1
            
        # Hash function
        lenMod = 2 ** (len(minimalPacket)-1)
        hashValue = lenMod * 518400000
        for i in range(len(minimalPacket)):
            tempHash = 1
            hashValue += 518400000 * (minimalPacket[i][0] + 1)
            tempHash *= lenMod * (minimalPacket[i][1] + 1)
            tempHash *= lenMod * (minimalPacket[i][2] + 1)
            tempHash *= lenMod * (minimalPacket[i][3] + 1)
            tempHash *= lenMod * (minimalPacket[i][4] + 1)
            hashValue += (tempHash * (i+1))
            
        #Here I'm splitting the collision resolution to deal with the case where the function is given a state list to check against
        #I'm doing this because the memory requirements are very large so I need a way to free up memory but still get hash values,
        #so in the freeSpace method I need new hash values for the used states to be hashed to, and for that I need a fresh state list.
        numCollisions = 0
        
        if not stateListToCheckAgainst:
            try:
                #Collision resolution
                while minimalPacket != self.stateList[hashValue][2]:
                    numCollisions += 1
                    hashValue += (numCollisions ** 2) # Quadratic collision resolution
                self.stateList[hashValue][1] += 1
            except KeyError:
                self.stateList[hashValue] = [deepcopy(state), 1, minimalPacket]
            return hashValue, numCollisions
        else:
            try:
                while minimalPacket != stateListToCheckAgainst[hashValue][2]:
                    numCollisions += 1
                    hashValue += (numCollisions ** 2) # Quadratic collision resolution
            except KeyError:
                stateListToCheckAgainst[hashValue] = [deepcopy(state), 1, minimalPacket]
            return hashValue, stateListToCheckAgainst
    
    
    def getMagAndDirFromIndex(self, index):
        magnitude = index // self.magAndDirModifier
        direction = (index % self.magAndDirModifier) * self.angleStepSize
        return magnitude, direction
    
    def reset(self):
        self.eps = 1.
    
    def doAction(self, state, action):
        currTeam = state[0][0]
        magnitude, direction = self.getMagAndDirFromIndex(action)
        
        distanceReward = 0
        for i in range(len(state)):
            if state[i][0] != state[0][0]: #They're on different teams
                distanceReward += state[i][7]
                
        state = utilities.adjustPacketForMove(state, magnitude, direction, self.Walls)
        
        i = 0
        deserter = False
        while i < len(state):
            if state[i][0] != state[0][0]: #They're on different teams
                distanceReward -= state[i][7] #Subtract new distances from old total
                if state[i][7] >= 100:
                    distanceReward -= state[i][1] #They've fled the battlefield! Penalize according to strength of the army
                    deserter = True #Remove the deserters from the battle
            i += 1
        distanceReward *= .01 #1% of distance closed is used as reward
        
        dmgArray = self.getDmgArray(state)
        if deserter:
            dmgArray[0] = state[0][1] #Deserters are killed on sight
        damageReward = (np.sum(dmgArray[1:]) * 1.75) - (dmgArray[0] * 1.)
        
        Reward = damageReward + distanceReward
        
        state, _, nextOpPos = utilities.adjustPacketForDamage(state, 0, dmgArray)
        Reward, endState, _ = utilities.getEndStateAndScoreAdjustment(state, Reward, currTeam, nextOpPos)
        if len(state) > 1:
            nextState = utilities.adjustPacketForRecursion(state, state[1])
        else:
            nextState = state
        
        return nextState, Reward, endState
    
    def greedyAction(self, state):
        moveProbability = np.ones(self.actionSpaceSize) * self.eps / self.actionSpaceSize
        bestMove = np.argmax(self.Q[state])
        moveProbability[bestMove] += (1. - self.eps)
        return moveProbability
    
    def softmaxAction(self, state):
        moveProbability = np.exp(self.Q[state]/(1500/self.stateList[state][1]))/np.sum(np.exp(self.Q[state]/(1500/self.stateList[state][1])))
        return moveProbability
    
    def sarsa(self, iterations, initialState):
        initialStateIndex, _ = self.getIndexForState(initialState)
        maxTurns = 5
        
        for i in range(iterations):
            currentState = self.stateList[initialStateIndex][0]
            self.stateList[initialStateIndex][1] += 1
            currentStateIndex = initialStateIndex
            
            moveProbability = self.softmaxAction(currentStateIndex)
            try:
                currentAction = np.random.choice(np.arange(self.actionSpaceSize), p = moveProbability)
            except ValueError:
                currentAction = np.argmax(moveProbability) # nan case
            
            turns = 0
            while True:
                nextState, Reward, endState = self.doAction(deepcopy(currentState), currentAction)
                
                sameTeam = True
                try:
                    if nextState[0][0] != currentState[0][0]:
                        sameTeam = False
                except IndexError:
                    #Enters here if all armies are dead
                    pass
                
                nextStateIndex, _ = self.getIndexForState(nextState)
                moveProbability = self.softmaxAction(nextStateIndex)
                
                try:
                    nextAction = np.random.choice(np.arange(self.actionSpaceSize), p = moveProbability)
                except ValueError:
                    nextAction = np.argmax(moveProbability) # nan case
                
                if sameTeam:
                    target = Reward + self.gamma * self.Q[nextStateIndex][nextAction] #If a teammate is moving next turn then add the reward from their move
                else:
                    target = Reward - self.gamma * self.Q[nextStateIndex][nextAction] #Otherwise subtract the reward of the opponent

                error = target - self.Q[currentStateIndex][currentAction]
                self.Q[currentStateIndex][currentAction] += self.alpha * error
                
                if endState or turns >= maxTurns:
                    break
                
                currentState = nextState
                currentStateIndex = nextStateIndex
                currentAction = nextAction
                turns += 1
        

    def get_move(self, packet, walls):
        self.loadWalls(walls)
        hashValue, _ = self.getIndexForState(packet)
        self.sarsa(5000, deepcopy(packet))
        i = 0
        while np.max(self.softmaxAction(hashValue)) < .95 and i < 10:# and self.stateList[nStateIndex][1] < 500):
            self.sarsa(1000, deepcopy(packet))
            i += 1
        action = np.argmax(self.Q[hashValue])
        mmag, mdir = self.getMagAndDirFromIndex(action)
        self.syncMemories(self.wallHash)
        return [mmag, mdir]
    
    def syncMemories(self, wallHash):
        path = 'sarsa/{0}/{0}DB'.format(wallHash)
        indexList = []
        for index in self.stateList:
            if np.max(self.softmaxAction(index)) > .1:
                indexList.append(index)   
        data = []
        for index in indexList:
            data.append([self.stateList[index][0], self.Q[index], self.stateList[index][1]])
        
        try:
            db = shelve.open(path, "n")
            db['data'] = data
            db.close()
        except KeyboardInterrupt:
            db.close()
            db = shelve.open(path, "n")
            db['data'] = data
            db.close()
        
    #This function sets the wall hash value and loads the Q-tables associated with the wall configuration
    def loadWalls(self, walls):
        wallHash = 1
        for wall in walls:
            for coord in wall:
                wallHash *= (coord[0] + coord[1] + 1)
        if wallHash != self.wallHash:
            self.loadMemories(wallHash)
            self.Walls = walls
            self.wallHash = wallHash
    
    def freeSpace(self, limit):
        print("\nFREEING SPACE")
        usedIndices = []
        for index in self.stateList:
            if np.max(self.softmaxAction(index)) > .1:
                usedIndices.append(index)
        print(len(usedIndices), "indices saved")
        newStateList = {}
        newQ = defaultdict(lambda: np.zeros(self.actionSpaceSize))
        for index in usedIndices:
            hashValue, _ = self.getIndexForState(self.stateList[index][0], newStateList)
            newStateList[hashValue] = self.stateList[index]
#            newStateList[hashValue][1] = self.stateList[index][1]
            newQ[hashValue] = self.Q[index]
            
        print("SPACE FREED. REDUCED MEMORY USE BY", (1 - (len(newStateList)/len(self.stateList)))*100, "%")
        del self.stateList
        del self.Q
        self.stateList = newStateList
        self.Q = newQ
        print("CHECKING INTEGRITY")
        for index in self.stateList:
            if self.stateList[index][1] == 1 or (np.argmax(self.Q[index]) == 0 and np.argmin(self.Q[index]) == 0):
                input("ERROR IN FREEING SPACE")
        print("DONE\n")
    
    ########## HELPER FUNCTIONS ##########
    
    def checkStalemate(self, state):
        if len(state) <= 1 or len(state) > 2:
            return False
        elif state[0][1] == state[1][1]:
            return True
        else:
            return False
                
    
    def checkCombatModifiers(self, unit1, unit2, walls):
            
            lineOfSightBlocked, _ = utilities.checkForIntersect(self.Walls, [unit1[4],unit1[5]], [unit2[4], unit2[5]])
            
            #Check if unit1 is attacking unit2
            if utilities.get_relative_direction(unit1, unit2) < utilities.get_relative_direction(unit2, unit1):
                Attacker = True
            else:
                Attacker = False

            dist = utilities.get_distance([unit1[4],unit1[5]], [unit2[4], unit2[5]])
            if unit1[2] > dist and unit2[2] > dist:
                #They're both in range of each other
                Range = False 
            else:
                Range = True #One outranges the other
                #Attacker is the one that outranges
                if unit1[2] > unit2[2]:
                    Attacker = True
                else:
                    Attacker = False
                    
            if not Attacker and utilities.get_relative_direction(unit1, unit2) > 30 and utilities.get_relative_direction(unit2, unit1) < 90:
                Bonus = True
            elif Attacker and utilities.get_relative_direction(unit2, unit1) > 30 and utilities.get_relative_direction(unit1, unit2) < 90:
                Bonus = True
            else:
                Bonus = False
                
            return lineOfSightBlocked, Attacker, Bonus, Range

    def calculateDamage(self, attDirection, attStr, defStr, Bonus, defOutOfRange):
            #Bonus is the attacking bonus for flanking
            if not Bonus:
                bonus = 0
            else:
                bonus = .75 * np.floor(attDirection / 15)# * (1 - (min(defHead, attDirection) / max(defHead, attDirection)))
            defDmgTaken = np.ceil(attStr / defStr + bonus)
            if not defOutOfRange:
                attDmgTaken = np.ceil(defStr / attStr - .5 * bonus) #Flanks are 75% effective at protecting attackers
            else:
                attDmgTaken = 0
            if attDmgTaken < 0:
                attDmgTaken = 0
            return defDmgTaken, attDmgTaken

    def getDmgArray(self, packet):
        #[Team, Str, fireRange, moveSpeed, x, y, heading, distance, direction]
        #  0     1       2         3       4  5     6        7          8
        dmgArray = np.zeros(len(packet))
        for i in range(1, len(packet)):
            if packet[i][0] == packet[0][0]:
                continue #Same team
            if packet[0][2] < packet[i][7] and packet[i][2] < packet[i][7]:
                continue #Both out of range
            lineOfSightBlocked, Attacker, Bonus, Range = self.checkCombatModifiers(packet[0], packet[i], self.Walls)
            if lineOfSightBlocked:
                continue
            if Attacker:
                attDirection = utilities.get_relative_direction(packet[i], packet[0])
                dmgDone, dmgTaken = self.calculateDamage(attDirection,
                                                         packet[0][1], packet[i][1],
                                                         Bonus, Range)
            else:
                attDirection = utilities.get_relative_direction(packet[0], packet[i])
                dmgTaken, dmgDone = self.calculateDamage(attDirection,
                                                         packet[i][1], packet[0][1],
                                                         Bonus, Range)
                
            dmgArray[i] += dmgDone
            dmgArray[0] += dmgTaken
            
        return dmgArray
    
    #Takes out all of the unnecessary information and rounds all floats to the nearest integer.
    #Preps packet for hashing to the state list
#    def minimizePacket(self, packet):
#        for i in range(len(packet)):
#            packet[i].pop(2) #Remove fireRange
#            packet[i].pop(2) #Remove moveSpeed
#            packet[i].pop(2) #Remove x-coord
#            packet[i].pop(2) #Remove y-coord
#            packet[i][1] = int(round(packet[i][1])) #Round strength (not necessary but keeps it neat)
#            packet[i][2] = int(round(packet[i][2])) #Round heading
#            packet[i][3] = int(round(packet[i][3])) #Round distance
#            packet[i][4] = int(round(packet[i][4])) #Round direction
#
#        return packet
    
    def minimizePacket(self, packet):
        for i in range(len(packet)):
            packet[i].pop(2) #Remove fireRange
            packet[i].pop(2) #Remove moveSpeed
            packet[i].pop(2) #Remove x-coord
            packet[i].pop(2) #Remove y-coord
            packet[i][1] = int(round(packet[i][1])) #Round strength (not necessary but keeps it neat)
            packet[i][2] = int(round(packet[i][2]/5)*5) % 360#Round heading
            packet[i][3] = int(round(packet[i][3]/2)*2) #Round distance
            packet[i][4] = int(round(packet[i][4]/5)*5) % 360#Round direction
        return packet

def shuffle(data):
    new = []
    while len(data) > 0:
        index = random.randint(0, len(data)-1)
        new.append(data[index])
        data.remove(data[index])
    return new

#random.seed(2640)

if __name__ == "__main__":
    S = SARSA()
    eps = [[.75, .75, .75, .75, .75, .75],
           [.5, .75, .75, .75, .75, .75],
           [.25, .5, .75, .75, .75, .75],
           [.1, .25, .5, .75, .75, .75],
           [.05, .1, .25, .5, .75, .75],
           [.01, .05, .1, .25, .5, .75]]
           
    iters = [500, 500, 500, 500, 500, 100]
    
    db = shelve.open('maxnet_data/maxnet_dataDB', 'r')
    data = db['data']
    db.close()
    data = shuffle(data[5000:7500])
    ttime = 0
    for i in range(len(data)):
        s = time.time()
        for j in range(len(eps)):
            packet = deepcopy(data[i][0])
            S.sarsa(iters[j], packet, eps[j])
        f = time.time()
        ttime += f-s
        atime = ttime/(i+1)
            
        print((i+1), "/", len(data), "|| Time to finish:", round((atime * (len(data) - (i+1)))/60,3), "mins")
        if (i+1) % 100 == 0:
            numQStatesUsable = 0
            S.freeSpace(1)
            for index in S.stateList:
                if S.stateList[index][1] >= 100:
                    numQStatesUsable += 1
            print("Usable Q Tables:", numQStatesUsable)
        if (i+1) % 500 == 0:
            S.freeSpace(5)
    indexList = []
    numQStatesUsable = 0
    for index in S.stateList:
        if S.stateList[index][1] >= 100:
            numQStatesUsable += 1
            indexList.append(index)
    print("Usable Q Tables:", numQStatesUsable)     
    data = []
    for index in indexList:
        data.append([S.stateList[index][0], S.Q[index], S.stateList[index][1]])
    db = shelve.open("sarsa/sarsaDB", "c")
    db['data'] = data
    db.close()











