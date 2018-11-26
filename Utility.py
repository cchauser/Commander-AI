import numpy as np

class Utility(object):
    def __init__(self):
        pass
    
    def get_relative_direction(self, unit1, unit2):
        if type(unit1) == list:
            unit1Heading = unit1[6]
            unit1Location = np.asarray([unit1[4], unit1[5]])
            unit2Location = np.asarray([unit2[4], unit2[5]])
        else:
            unit1Heading = unit1.heading
            unit1Location = unit1.location
            unit2Location = unit2.location

        zeroedLocation = unit2Location - unit1Location #Translate coordinates as though unit1Location is 0,0
        angle = np.arctan2(zeroedLocation[0], zeroedLocation[1]) * 180 / np.pi
        angle = np.abs(unit1Heading - angle)
        if angle > 180:
            angle = np.abs(360 - angle)
        
        return angle
    
    def get_absolute_direction(self, unit1, unit2):
        if type(unit1) == list:
            unit1Location = np.asarray([unit1[4], unit1[5]])
            unit2Location = np.asarray([unit2[4], unit2[5]])
        else:
            unit1Location = unit1.location
            unit2Location = unit2.location

        zeroedLocation = unit2Location - unit1Location #Translate coordinates as though unit1Location is 0,0
        angle = np.arctan2(zeroedLocation[0], zeroedLocation[1]) * 180 / np.pi
        if angle < 0:
            angle += 360
        return angle

    def get_distance(self, unit1, unit2):
        if type(unit1) == list:
            unit1Location = np.asarray([unit1[4], unit1[5]])
            unit2Location = np.asarray([unit2[4], unit2[5]])
        else:
            unit1Location = unit1.location
            unit2Location = unit2.location

        distance = np.linalg.norm(unit1Location - unit2Location)
        return distance


    def printPacket(self, packet):
        for unit in packet:
            print(unit)

    def parseHumanMove(self, inputString):
        inputString = inputString.split()
        inputString[0] = int(inputString[0])
        inputString[1] = int(inputString[1])
        return inputString

    def get_team_strengths(self, teamArray):
        strengths = np.zeros(len(teamArray))
        for i in range(len(teamArray)):
            for unit in teamArray[i].get_armies():
                strengths[i] += unit.strength
        return strengths
    
    def get_dXdY(self, magnitude, direction, moveSpeed):
        distance = (min(magnitude, 10) / 10) * moveSpeed
        direction = direction % 360
        
        tempHead = (direction) * np.pi / 180 #This is for finding length of the x,y changes from a right triangle
        dX = distance * np.sin(tempHead)
        dY = distance * np.cos(tempHead)

        return int(dX), int(dY)
    
    def adjustPacketForMove(self, packet, magnitude, direction):
        dX, dY = self.get_dXdY(magnitude, direction, packet[0][3])
        packet[0][4] += dX
        packet[0][5] += dY
        packet[0][6] = direction
        for i in range(1, len(packet)):
            packet[i][-2] = self.get_distance(packet[0], packet[i])
            packet[i][-1] = self.get_absolute_direction(packet[0], packet[i])
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
                continue #Army is dead and deleted, go to next army without incrementing i
            if currTeam != packet[i][0] and i < nextOpponentPositionInPacket:
                nextOpponentPositionInPacket = i
            i += 1
        return packet, Score, nextOpponentPositionInPacket
    
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

    def adjustPacketForRecursion(self, packet, nextOpponent):
        #Reorganize the packet so that the next unit to move is at the top
        packet.append(packet[0])
        packet.remove(packet[0])
        packet.remove(nextOpponent)
        packet.insert(0, nextOpponent)
    
        packet[0][-2] = 0
        packet[0][-1] = 0
        for i in range(1, len(packet)):
            packet[i][-2] = self.get_distance(packet[0], packet[i])
            packet[i][-1] = self.get_absolute_direction(packet[0], packet[i])
        return packet