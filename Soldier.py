import numpy as np

from Unit import Unit
from Utility import Utility

utilities = Utility()

class Soldier(Unit):
    def __init__(self, unitID, unitTeam, strength, location, heading):
        #location is nparray [lat,long]
        #Heading is degrees, 0 being north, 90 being east
        self.__unitID = unitID
        self.__unitTeam = unitTeam
        
        self.set_strength(strength)

        #Soldier move speed will be based on their strength (number of "soldiers")
        baseMoveSpeed = 20 #Base move speed is for 20 soldiers
        
        moveSpeed = baseMoveSpeed + (((20 // strength) // 2) * 10)
        moveSpeed = int(moveSpeed)
        
        self.set_moveSpeed(moveSpeed)

        fireRange = 40 #Fire range for soldiers   
        self.set_fireRange(fireRange)

        self.set_location(location)
        self.set_heading(heading)

    
    
    def set_moveSpeed(self, moveSpeed):
        self.moveSpeed = moveSpeed

    def get_moveSpeed(self):
        return self.moveSpeed

    def get_unitID(self):
        return self.__unitID
    
    def get_unitTeam(self):
        return self.__unitTeam

    def get_packet(self):
        return [self.__unitTeam, self.strength, self.fireRange, self.moveSpeed,
                self.location[0], self.location[1], self.heading]

    def move(self, magnitude, direction, walls):
        distance = (min(magnitude, 10) / 10) * self.moveSpeed
        direction = direction % 360
        
        tempHead = (direction) * np.pi / 180 #This is for finding length of the x,y changes from a right triangle
        dX = distance * np.sin(tempHead)
        dY = distance * np.cos(tempHead)

        dLocation = [int(dX), int(dY)]
        
        #Prevent moving through walls
        #Other units may define this differently so it's handled within the Soldier class
        #All different unit sub-classes need to handle walls themselves
        proposedLocation = self.location + dLocation
        intersect, intersectLocation = utilities.checkForIntersect(walls, self.location, proposedLocation, True)
        print(intersect, intersectLocation)
        if intersect:
            distance = utilities.get_distance(self.location, intersectLocation) - 3 #leaves a 3 unit buffer between the unit and the wall
            print(distance)
            dX = distance * np.sin(tempHead)
            dY = distance * np.cos(tempHead)
            dLocation = [int(dX), int(dY)]
            
        
        print(proposedLocation)
        self.heading = direction
        self.location += dLocation
        print(self.location)






        
