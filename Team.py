import numpy as np

from Soldier import Soldier
from random import randint

class Team(object):
    def __init__(self, teamID, size, spawnY, unitID_counter, randomSpawn = 0):
        self.armies = []
        negOrPos = spawnY/spawnY

        if randomSpawn == 0:
            spawn = np.asarray([0,spawnY])
        else:
            spawn = np.asarray([0,0])
            
        if spawnY > 0:
            spawnHeading = 180
        else:
            spawnHeading = 0
        

        possibleStrengths = [5,10,20]
        soldierID = 0
        for i in range(size):
            newSoldierStrength = possibleStrengths[randint(0,2)]
            if randomSpawn == 0:
                spawnCoordinates = spawn + [randint(-30,30), randint(0,30) * negOrPos]
            else:
                spawnCoordinates = spawn + [randint(-30,30), randint(-30,30)]
                if spawnCoordinates[1] <= 0:
                    spawnHeading = 0
                else:
                    spawnHeading = 180

            self.armies.append(Soldier(unitID_counter, teamID, newSoldierStrength, spawnCoordinates, spawnHeading))
            unitID_counter += 1

    def get_armies(self):
        return self.armies

    def __getitem__(self, index):
        return self.armies[index]
