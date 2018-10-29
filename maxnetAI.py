# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:37:54 2018

@author: Cullen
"""
import numpy as np
import random
import shelve

from copy import deepcopy
from LSTM import LSTM
from Utility import Utility

utilities = Utility()

class maxnetAI(object):
    def __init__(self, magModel = 'magBrain', dirModel = 'dirBrain', recursionLimit = 2, calibrationSet = None):
        self.recursionLimit = recursionLimit
        self.magWindow = -1
        self.dirWindow = -1
        
        print("Loading magBrain")
        self.magBrain = LSTM(magModel, 1, 'mag')
        print("DONE\n\nLoading dirBrain")
        self.dirBrain = LSTM(dirModel, 1, 'dir')
        print("DONE")
        
        if calibrationSet:
            if type(calibrationSet) == list:
                self.calibrateMax(packetList = calibrationSet)
            else:
                self.calibrateMax(databaseName = calibrationSet)
            
        
    def calibrateMax(self, databaseName = None, packetList = None):
        calibrationSet = []
        if databaseName:
            db = shelve.open("{0}/{0}DB".format(databaseName), "r")
            data = db['data']
            db.close()
            for i in range(10):
                calibrationSet.append(data[random.randint(0,len(data))][0])
        elif packetList:
            calibrationSet = packetList
        else:
            return False
        
        print("CALIBRATING ON {} POINTS".format(len(calibrationSet)))
        
        magTotal = 0
        dirTotal = 0

        for packet in calibrationSet:
            nnetMag = round(self.magBrain.nnet_move(packet)[0])
            nnetDir = round(self.dirBrain.nnet_move(packet)[0])
            
            minmaxMove = self.maxMove(packet, 1)[1:]
            
            magTotal += max(nnetMag, minmaxMove[0]) - min(nnetMag, minmaxMove[0])
            dirTotal += max(nnetDir, minmaxMove[1]) - min(nnetDir, minmaxMove[1])
            
        self.magWindow = int(np.ceil(magTotal / len(calibrationSet))) + 1
        self.dirWindow = int(np.ceil(dirTotal / len(calibrationSet))/10) + 3
        
        print(self.magWindow, self.dirWindow)
                
        
    def get_move(self, packet):
        move = self.maxMove(packet, self.recursionLimit)
        return move[1:]
            
    def maxMove(self, originalPacket, lookAheadLimit, recursionStep = 0):
        bestMove = [-np.inf, 0, 0]
        if self.magWindow != -1:
            nnetMag = int(round(self.magBrain.nnet_move(originalPacket)[0]))
            magRange = (max(nnetMag-self.magWindow,0), min(11,nnetMag+self.magWindow+1))
        else:
            magRange = (0,11)
        if self.dirWindow != -1:
            nnetDir = int(round(self.dirBrain.nnet_move(originalPacket)[0]))
            dirRange = (nnetDir-self.dirWindow, nnetDir+self.dirWindow+1)
        else:
            dirRange = (0,36)
            
        
        for magnitude in range(magRange[0], magRange[1]):
            for direction in range(dirRange[0], dirRange[1]):
                endState = False
                packet = deepcopy(originalPacket)
                
                direction = (direction % 36) * 10
                dX, dY = self.get_dXdY(magnitude, direction, packet[0][3])
                packet[0][4] += dX
                packet[0][5] += dY
                packet[0][6] = direction
                for i in range(1, len(packet)):
                    packet[i][-2] = utilities.get_distance(packet[0], packet[i])
                    packet[i][-1] = utilities.get_relative_direction(packet[0], packet[i])

                if recursionStep < lookAheadLimit:
                    Score, dTaken, dmgArray = self.evalMove(packet)
                    #Update army strengths based on simulated damage. Get next opponent while we're iterating through
                    i = 0
                    nextOpponentInPacket = len(packet) #Position of next opponent in packet
                    currTeam = packet[0][0]
                    while i < len(packet):
                        packet[i][1] -= dmgArray[i]
                        if packet[i][1] <= 0:
                            if i == 0:
                                Score -= 10 #Penalize for losing an army
                            else:
                                Score += 10 #Reward for destroying army
                            packet.remove(packet[i])
                            dmgArray = np.delete(dmgArray, i)
                            continue #Army is dead and deleted, go to next army
                        if currTeam != packet[i][0] and i < nextOpponentInPacket:
                            nextOpponentInPacket = i
                        i += 1
                    if nextOpponentInPacket < len(packet):
                        nextOpponent = packet[nextOpponentInPacket]
                    elif len(packet) > 0:
                        #No opponents left, win game with this move
                        Score += 100
                        #I could return the move here but seeing as this is a military game, the best course is to make sure that the
                        #move returned also minimizes losses of my own side.
                        endState = True
                    else:
                        #Everybody is dead, no score increase but set as endState
                        endState = True
                        
                    #Prune
                    if Score < bestMove[0]:
                        continue
                    elif not endState:
                        #Reorganize the packet so that the next unit to move is at the top
                        packet.append(packet[0])
                        packet.remove(packet[0])
                        packet.remove(nextOpponent)
                        packet.insert(0, nextOpponent)
    
                        packet[0][-2] = 0
                        packet[0][-1] = 0
                        for i in range(1, len(packet)):
                            packet[i][-2] = utilities.get_distance(packet[0], packet[i])
                            packet[i][-1] = utilities.get_relative_direction(packet[0], packet[i])
                           
                        #Subtract the opponent's best move from our best move's score
                        Score -= self.maxMove(packet, lookAheadLimit, recursionStep + 1)[0]
                    
                #If this is a leaf node there's no need to prepare for recursion. Just get the score
                else:
                    Score, dTaken, dmgArray = self.evalMove(packet)
                    
                    #Update army strengths based on simulated damage. Get next opponent while we're iterating through
                    i = 0
                    nextOpponentInPacket = len(packet)
                    currTeam = packet[0][0]
                    while i < len(packet):
                        packet[i][1] -= dmgArray[i]
                        if packet[i][1] <= 0:
                            if i == 0:
                                Score -= 10 #Penalize for losing an army
                            else:
                                Score += 10 #Reward for destroying army
                            #Don't delete here because this is a leaf node so just need to iterate through
                            i += 1
                            continue #Army is dead, go to next
                        if currTeam != packet[i][0] and i < nextOpponentInPacket:
                            nextOpponentInPacket = i
                        i += 1
                    if nextOpponentInPacket == len(packet) and len(packet) > 0:
                        #No opponents left, win game with this move
                        Score += 100
                        #I could return the move here but seeing as this is a military game, the best course is to make sure that the
                        #move returned also minimizes losses of my own side.
                        endState = True

                if Score > bestMove[0]:
                    bestMove = [Score, magnitude, direction]


        return bestMove
    
    
    
    
    def get_dXdY(self, magnitude, direction, moveSpeed):
        distance = (min(magnitude, 10) / 10) * moveSpeed
        direction = direction % 360
        
        tempHead = (direction) * np.pi / 180 #This is for finding length of the x,y changes from a right triangle
        dX = distance * np.sin(tempHead)
        dY = distance * np.cos(tempHead)

        return int(dX), int(dY)

    def check_Attacker_Bonus_Range(self, unit1, unit2):
        #Check if unit1 is attacking unit2
        if utilities.get_relative_direction(unit1, unit2) < utilities.get_relative_direction(unit2, unit1):
            Attacker = True
        else:
            Attacker = False
    
        #Just assume they can attack each other
        Range = False

        if not Attacker and utilities.get_relative_direction(unit1, unit2) > 30 and utilities.get_relative_direction(unit2, unit1) < 90:
            Bonus = True
        elif Attacker and utilities.get_relative_direction(unit2, unit1) > 30 and utilities.get_relative_direction(unit1, unit2) < 90:
            Bonus = True
        else:
            Bonus = False

        return Attacker, Bonus, Range

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

    def evalMove(self, packet):
        #[Team, Str, fireRange, moveSpeed, x, y, heading, distance, direction]
        #  0     1       2         3       4  5     6        7          8
        dmgArray = np.zeros(len(packet))
        for i in range(1, len(packet)):
            if packet[i][0] == packet[0][0]:
                continue
            Attacker, Bonus, Range = self.check_Attacker_Bonus_Range(packet[0], packet[i])
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
            dmgDone = min(dmgDone, dmgDone * (40/packet[i][7])) + min(1, .5 / packet[i][7])
            dmgTaken = min(dmgTaken, dmgTaken * (40/packet[i][7]))
            
            dmgArray[i] += dmgDone
            dmgArray[0] += dmgTaken
##        if dDone > 0:
##            print("DAMAGE CALC", dDone - dTaken)
        Score =  np.sum(dmgArray[1:]) - dmgArray[0]
        return Score, np.sum(dmgArray[1:]), dmgArray