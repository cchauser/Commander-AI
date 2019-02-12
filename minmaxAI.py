from copy import deepcopy
import numpy as np
from Utility import Utility
from AI import AI

utilities = Utility()

class minmaxAI(AI):
    def __init__(self):
        pass

    def get_dXdY(self, magnitude, direction, moveSpeed):
        distance = (min(magnitude, 10) / 10) * moveSpeed
        direction = direction % 360
        
        tempHead = (direction) * np.pi / 180 #This is for finding length of the x,y changes from a right triangle
        dX = distance * np.sin(tempHead)
        dY = distance * np.cos(tempHead)

        return int(dX), int(dY)

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

    def evalMove(self, packet):
        #[Team, Str, fireRange, moveSpeed, x, y, heading, distance, direction]
        #  0     1       2         3       4  5     6        7          8
        dmgArray = np.zeros(len(packet))
        for i in range(1, len(packet)):
            if packet[i][0] == packet[0][0]:
                continue
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
            dmgDone = min(dmgDone, dmgDone * (30/packet[i][7])) + min(1, 10 / packet[i][7])
            dmgTaken = min(dmgTaken, dmgTaken * (40/packet[i][7]))
            
            dmgArray[i] += dmgDone
            dmgArray[0] += dmgTaken
        Score =  np.sum(dmgArray[1:]) - dmgArray[0]
        return Score, dmgArray
    
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
    
    def get_move(self, packet, walls):
        self.Walls = walls
        return self.minmaxMove(packet)

    def minmaxMove(self, originalPacket, lookAheadLimit=1, recursionStep = 0):
        currTeam = originalPacket[0][0]
        
        packet = deepcopy(originalPacket)
        try:
            magnitude, direction = self.getAggressiveMove(packet)
        except IndexError:
            return [0, 0, 0]
        
        packet = utilities.adjustPacketForMove(packet,magnitude,direction,self.Walls)
        Score, dmgArray = self.evalMove(packet)
        packet, Score, nextOpponentPositionInPacket = utilities.adjustPacketForDamage(packet, Score, dmgArray)
        Score, endState, nextOpponent = utilities.getEndStateAndScoreAdjustment(packet, Score, currTeam, nextOpponentPositionInPacket)
        if not endState and recursionStep < lookAheadLimit:
            packet = utilities.adjustPacketForRecursion(packet, nextOpponent)
            Score -= self.minmaxMove(packet, lookAheadLimit, recursionStep + 1)[2]
        bestMove = [magnitude, direction, Score]
        
        for magnitude in range(0, 11):
            for direction in range(0, 36):
                endState = False
                packet = deepcopy(originalPacket)
                
                direction = direction * 10
                
                packet = utilities.adjustPacketForMove(packet,magnitude,direction,self.Walls)
                
                if recursionStep < lookAheadLimit:
                    Score, dmgArray = self.evalMove(packet)
                    packet, Score, nextOpponentPositionInPacket = utilities.adjustPacketForDamage(packet, Score, dmgArray)
                    
                    Score, endState, nextOpponent = utilities.getEndStateAndScoreAdjustment(packet, Score, currTeam, nextOpponentPositionInPacket)
                        
                    #Prune
                    if Score < bestMove[2]:
                        continue
                    elif not endState:
                        packet = utilities.adjustPacketForRecursion(packet, nextOpponent)
                        
                        #Subtract the opponent's best move from our best move's score
                        Score -= self.minmaxMove(packet, lookAheadLimit, recursionStep + 1)[2]
                    
                #If this is a leaf node there's no need to prepare for recursion. Just get the score
                else:
                    Score, dmgArray = self.evalMove(packet)
                    
                    packet, Score, _ = utilities.adjustPacketForDamage(packet, Score, dmgArray)
                    
                    Score, _, _ = utilities.getEndStateAndScoreAdjustment(packet, Score, currTeam, nextOpponentPositionInPacket) # Underscores are unused variables

                if Score > bestMove[2]:
                    bestMove = [magnitude, direction, Score]


        return bestMove
    
    
    
    
    
    
    
    
    
    