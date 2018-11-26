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
    def __init__(self, magModel = 'magBrain', dirModel = 'dirBrain', recursionLimit = 1, angleStepSize = 10, calibrationSet = None):
        self.recursionLimit = recursionLimit
        self.angleStepSize = angleStepSize
        self.magWindow = 3
        self.dirWindow = 7
        
        print("Loading magBrain")
        self.magBrain = LSTM(magModel, 1, 'mag')
        print("DONE\n\nLoading dirBrain")
        self.dirBrain = LSTM(dirModel, 1, 'dir')
        print("DONE")
        
        if calibrationSet and (self.magWindow == -1 and self.dirWindow == -1):
            if type(calibrationSet) == list:
                self.calibrateWindow(packetList = calibrationSet)
            else:
                self.calibrateWindow(databaseName = calibrationSet)
            
        
    def calibrateWindow(self, databaseName = None, packetList = None):
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
            nnetDir = round(self.dirBrain.nnet_move(packet))
            nnetMag = round(self.magBrain.nnet_move(packet, heading = nnetDir))
            
            minmaxMove = self.maxMove(packet, 1)[1:]
            
            magTotal += max(nnetMag, minmaxMove[0]) - min(nnetMag, minmaxMove[0])
            dirTotal += max(nnetDir, minmaxMove[1]) - min(nnetDir, minmaxMove[1])
            
        self.magWindow = int(np.ceil(magTotal / len(calibrationSet)))
        self.dirWindow = int(np.ceil(dirTotal / len(calibrationSet))/self.angleStepSize)
        
        print(self.magWindow, self.dirWindow)
                
        
    def get_move(self, packet):
        move = self.maxMove(packet, self.recursionLimit)
        return move[1:]
    
    def adjustPacketForMove(self, packet, magnitude, direction):
        
        dX, dY = self.get_dXdY(magnitude, direction, packet[0][3])
        packet[0][4] += dX
        packet[0][5] += dY
        packet[0][6] = direction
        for i in range(1, len(packet)):
            packet[i][-2] = utilities.get_distance(packet[0], packet[i])
            packet[i][-1] = utilities.get_absolute_direction(packet[0], packet[i])
        return packet
    
    def adjustPacketForDamage(self, packet, Score, dmgArray):
        #Update army strengths based on simulated damage. Get next opponent while we're iterating through
        i = 0
        nextOpponentPositionInPacket = len(packet) #Position of next opponent in packet
        currTeam = packet[0][0]
        currArmyDead = False
        while i < len(packet):
            packet[i][1] -= dmgArray[i]
            if packet[i][1] <= 0:
                if i == 0 and not currArmyDead:
                    Score -= 10 #Penalize for losing an army
                else:
                    Score += 10 #Reward for destroying army
                packet.remove(packet[i])
                dmgArray = np.delete(dmgArray, i)
                continue #Army is dead and deleted, go to next army
            if currTeam != packet[i][0] and i < nextOpponentPositionInPacket:
                nextOpponentPositionInPacket = i
            i += 1
        return packet, Score, nextOpponentPositionInPacket
    
    def adjustPacketForRecursion(self, packet, nextOpponent):
        #Reorganize the packet so that the next unit to move is at the top
        packet.append(packet[0])
        packet.remove(packet[0])
        packet.remove(nextOpponent)
        packet.insert(0, nextOpponent)
    
        packet[0][-2] = 0
        packet[0][-1] = 0
        for i in range(1, len(packet)):
            packet[i][-2] = utilities.get_distance(packet[0], packet[i])
            packet[i][-1] = utilities.get_absolute_direction(packet[0], packet[i])
        return packet
    
    def getEndStateAndScoreAdjustment(self, packet, Score, currTeam, nextOpPosPack):
        currTeamDead = True
        endState = False
        nextOpponent = -1
        for unit in packet:
            if unit[0] == currTeam:
                currTeamDead = False #Still have an army alive on the team
                break
        
        if currTeamDead:
            Score -= 100
            endState = True
        elif nextOpPosPack < len(packet):
            nextOpponent = packet[nextOpPosPack]
        elif len(packet) > 0:
            #No opponents left, win game with this move
            Score += 100
            #I could return the move here but seeing as this is a military game, the best course is to make sure that the
            #move returned also minimizes losses of my own side.
            endState = True
        else:
            #Everybody is dead, no score increase but set as endState
            endState = True
        return Score, endState, nextOpponent
    
    #Returns a similar move to dumb ai, but a little smarter
    def getAggressiveMove(self, packet):
        currTeam = packet[0][0]
        target = [0, np.inf] #Need a sample container for comparison.
        #Pick out the weakest enemy and charge them
        for unit in packet:
            if currTeam != unit[0] and target[1] > unit[1]:
                target = unit
        direction = target[8]
        distance = target[7]-10 #10 unit buffer
        for mag in range(11):
            newDist = (mag/10) * packet[0][3]
            if newDist > distance:
                if mag > 0:
                    mag -= 1
                break
        return mag, direction
                
        
    def maxMove(self, originalPacket, lookAheadLimit, recursionStep = 0):
        currTeam = originalPacket[0][0]
        
        packet = deepcopy(originalPacket)
        magnitude, direction = self.getAggressiveMove(packet)
        
        packet = self.adjustPacketForMove(packet,magnitude,direction)
        Score, dmgArray = self.evalMove(packet)
        packet, Score, nextOpponentPositionInPacket = self.adjustPacketForDamage(packet, Score, dmgArray)
        Score, endState, nextOpponent = self.getEndStateAndScoreAdjustment(packet, Score, currTeam, nextOpponentPositionInPacket)
        if not endState and recursionStep < lookAheadLimit:
            packet = self.adjustPacketForRecursion(packet, nextOpponent)
            Score -= self.maxMove(packet, lookAheadLimit, recursionStep + 1)[0]
        bestMove = [Score, magnitude, direction]
            
        
        if self.magWindow != -1 or self.dirWindow != -1:
            nnetDir = int(round(self.dirBrain.nnet_move(originalPacket)/self.angleStepSize))
            dirRange = (nnetDir-self.dirWindow, nnetDir+self.dirWindow+1)
            
            nnetMag = int(round(self.magBrain.nnet_move(originalPacket, heading = nnetDir*self.angleStepSize)))
            magRange = (max(nnetMag-self.magWindow,0), min(11,nnetMag+self.magWindow+1))
        else:
            dirRange = (0,360/self.angleStepSize)
            magRange = (0,11)            
        
        
        for magnitude in range(magRange[0], magRange[1]):
            for direction in range(dirRange[0], dirRange[1]):
                endState = False
                packet = deepcopy(originalPacket)
                
                direction = (direction % (360 / self.angleStepSize)) * self.angleStepSize
                
                packet = self.adjustPacketForMove(packet,magnitude,direction)
                
                if recursionStep < lookAheadLimit:
                    Score, dmgArray = self.evalMove(packet)
                    packet, Score, nextOpponentPositionInPacket = self.adjustPacketForDamage(packet, Score, dmgArray)
                    
                    Score, endState, nextOpponent = self.getEndStateAndScoreAdjustment(packet, Score, currTeam, nextOpponentPositionInPacket)
                        
                    #Prune
                    if Score < bestMove[0]:
                        continue
                    elif not endState:
                        packet = self.adjustPacketForRecursion(packet, nextOpponent)
                        
                        #Subtract the opponent's best move from our best move's score
                        Score -= self.maxMove(packet, lookAheadLimit, recursionStep + 1)[0]
                    
                #If this is a leaf node there's no need to prepare for recursion. Just get the score
                else:
                    Score, dmgArray = self.evalMove(packet)
                    
                    packet, Score, _ = self.adjustPacketForDamage(packet, Score, dmgArray)
                    
                    Score, _, __ = self.getEndStateAndScoreAdjustment(packet, Score, currTeam, nextOpponentPositionInPacket) # Underscores are unused variables

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
#        bonusScore = 0
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
            if packet[i][7] > packet[0][2]:
                dmgDone = dmgDone * 30/packet[i][7] + min(1, 10 / packet[i][7])
                dmgTaken = dmgTaken * (40/packet[i][7])
            
            dmgArray[i] += dmgDone
            dmgArray[0] += dmgTaken
#            bonusScore += ((Attacker * Bonus) * 2) - (Bonus - Attacker)
        Score =  np.sum(dmgArray[1:]) - dmgArray[0]# + bonusScore
        return Score, dmgArray