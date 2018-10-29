import numpy as np

class Unit(object):
    def __init__(self, strength, fireRange, location, heading):
        self.set_strength(strength)
        self.set_fireRange(fireRange)
        self.set_location(location)
        self.set_heading(heading)

    def set_location(self, location):
        self.location = location

    def get_location(self):
        return self.location

    def set_heading(self, heading):
        heading = heading % 360 #If heading is 340, and instructed to turn right 45, this will correct that
        self.heading = heading

    def get_heading(self):
        return self.heading

    def set_strength(self, strength):
        self.strength = strength
        
    def set_fireRange(self, fireRange):
        self.fireRange = fireRange

    def get_strength(self):
        return self.strength

    def get_fireRange(self):
        return self.fireRange

    #returns direction (in degrees) of unit2 from unit1
    def get_direction(self, unit2):
        unit1Heading = self.heading
        unit1Location = self.location
        unit2Location = unit2.location

        zeroedLocation = unit2Location - unit1Location #Translate coordinates as though unit1Location is 0,0
        angle = np.arctan2(zeroedLocation[0], zeroedLocation[1]) * 180 / np.pi
        angle = np.abs(unit1Heading - angle)
        if angle > 180:
            angle = 360 - angle
        
        return angle

    def get_distance(self, unit2):
        unit1Location = self.location
        unit2Location = unit2.location

        distance = np.linalg.norm(unit1Location - unit2Location)
        return distance
