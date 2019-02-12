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

    def get_distance(self, point1, point2):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        distance = np.linalg.norm(point1 - point2)
        return distance
    
    def getIntersectLocation(self, wall, unitLocations):
        #cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html
        du = unitLocations[1] - unitLocations[0]
        dw = wall[1] - wall[0]
        dp = unitLocations[0] - wall[0]
        dup = np.asarray([-du[1], du[0]])
        denom = np.dot(dup, dw)
        num = np.dot(dup, dp)
        
        i = (num/denom.astype(float)) * dw + wall[0]
        
        return i
    
    def onSegment(self, point1, point2, point3):
        if point2[0] <= max(point1[0], point3[0]) and point2[0] >= min(point1[0], point3[0]) and point2[1] <= max(point1[1], point3[1]) and point2[1] >= min(point1[1], point3[1]):
            return True
        return False
    
    def orientation(self, point1, point2, point3):
        orientation = (point2[1] - point1[1]) * (point3[0] - point2[0]) - (point2[0] - point1[0]) * (point3[1] - point2[1])
        
        if orientation == 0:
            return 0
        elif orientation > 0:
            return 1
        else:
            return 2

    def checkForIntersect(self, walls, point1, point2, verbose = False):
        unitLocations = np.asarray([point1, point2])
        intersectLocations = []
        for wall in walls:
            w = np.asarray(wall)
            
            o1 = self.orientation(unitLocations[0], unitLocations[1], w[0])
            o2 = self.orientation(unitLocations[0], unitLocations[1], w[1])
            o3 = self.orientation(w[0], w[1], unitLocations[0])
            o4 = self.orientation(w[0], w[1], unitLocations[1])
            
            if o1 != o2 and o3 != o4:
                intersectLocations.append(self.getIntersectLocation(w, unitLocations))
        if len(intersectLocations) > 0:    
            minDist = np.inf
            for location in intersectLocations:
                distance = self.get_distance(location, point1)
                if verbose:
                    print(minDist, distance, point1, location)
                if distance < minDist:
                    closestIntersect = location
                    minDist = distance
            return True, closestIntersect
        else:
            return False, None

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
    
    def adjustPacketForMove(self, packet, magnitude, direction, walls):
        currentLocation = np.asarray([packet[0][4], packet[0][5]])
        dX, dY = self.get_dXdY(magnitude, direction, packet[0][3])
        proposedLocation = currentLocation + [dX, dY]
        intersect, intersectLocation = self.checkForIntersect(walls, currentLocation, proposedLocation)
        if intersect:
            distance = self.get_distance(currentLocation, intersectLocation) - 3 #leave a 3 unit buffer between the unit and the wall
            tempHead = direction * np.pi / 180
            dX = distance * np.sin(tempHead)
            dY = distance * np.cos(tempHead)
        packet[0][4] += dX
        packet[0][5] += dY
        packet[0][6] = direction
        for i in range(1, len(packet)):
            packet[i][-2] = self.get_distance([packet[0][4], packet[0][5]], [packet[i][4], packet[i][5]])
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
            Score -= 25
            endState = True
        elif nextOpPosPack < len(packet):
            nextOpponent = packet[nextOpPosPack]
        elif len(packet) > 0:
            #No opponents left, win game with this move
            Score += 25
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
            packet[i][-2] = self.get_distance([packet[0][4], packet[0][5]], [packet[i][4], packet[i][5]])
            packet[i][-1] = self.get_absolute_direction(packet[0], packet[i])
        return packet