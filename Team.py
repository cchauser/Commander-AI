import numpy as np

from Soldier import Soldier
from random import randint

class Team(object):
    def __init__(self, teamID, size, spawn, unitID_counter, randomSpawn = 0):
        self.armies = []
        if spawn[1] < 0:
            negOrPos = -1
        else:
            negOrPos = 1

        if randomSpawn != 0:
            spawn = np.asarray([0,0])
        else: 
            spawn = np.asarray(spawn)
        
        possibleStrengths = [5,10,20]
        for i in range(size):
            newSoldierStrength = possibleStrengths[randint(0,2)]
            if randomSpawn == 0:
                spawnCoordinates = spawn + [randint(-60,60), randint(0,20) * negOrPos]
            else:
                spawnCoordinates = spawn + [randint(-60,60), randint(-20,20)]

            self.armies.append(Soldier(unitID_counter, teamID, newSoldierStrength, spawnCoordinates, 0)) #TODO: Fix deprecated spawnHeading
            unitID_counter += 1

    def get_armies(self):
        return self.armies

    def __getitem__(self, index):
        return self.armies[index]
