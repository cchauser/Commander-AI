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
