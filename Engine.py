import numpy as np
import random

from Team import Team
from graphicsEngine import graphicsEngine
from minmaxAI import minmaxAI
from dumbAI import dumbAI
from randAI import randAI
from neurnetAI import neurnetAI
from maxnetAI import maxnetAI
from Utility import Utility

utilities = Utility()

class Engine(object):
    def __init__(self, redTeamSize, blueTeamSize, redController = "minmax", blueController = "minmax", nnet_train = True):
        self.graphics = graphicsEngine()
        self.redController = redController
        self.blueController = blueController
        self.nnet_train = nnet_train
        self.t = 1 # This is for the neural network training
        
        self.redTeam = Team(0, redTeamSize, 20, 0)
        self.blueTeam = Team(1, blueTeamSize, -20, redTeamSize)

        self.turnOrder = []
        redC = 0
        blueC = 0
        while len(self.turnOrder) < redTeamSize + blueTeamSize:
            if redC < len(self.redTeam.armies):
                self.turnOrder.append(self.redTeam[redC])
                redC += 1
            if blueC < len(self.blueTeam.armies):
                self.turnOrder.append(self.blueTeam[blueC])
                blueC += 1

        self.minmaxAI = minmaxAI()
        self.dumbAI = dumbAI()
        self.randAI = randAI()
        if 'neurnet' in [redController, blueController]:
            self.neurnetAI = neurnetAI(11, 36)
        if 'maxnet' in [redController, blueController]:
            self.maxnetAI = maxnetAI(calibrationSet = 'good_data')
#        self.graphics.drawState(self.redTeam, self.blueTeam)

    def reset(self, redTeamSize, blueTeamSize, redController = "minmax", blueController = "minmax", allowRandom = False):
        self.redController = redController
        self.blueController = blueController
        
        if allowRandom:
            randomSpawn = random.randint(0,2) #Random spawns occur when value != 0
        else:
            randomSpawn = 0
        self.redTeam = Team(0, redTeamSize, 20, 0, randomSpawn)
        self.blueTeam = Team(1, blueTeamSize, -20, redTeamSize, randomSpawn)

        self.turnOrder = []
        redC = 0
        blueC = 0
        while len(self.turnOrder) < redTeamSize + blueTeamSize:
            if redC < len(self.redTeam.armies):
                self.turnOrder.append(self.redTeam[redC])
                redC += 1
            if blueC < len(self.blueTeam.armies):
                self.turnOrder.append(self.blueTeam[blueC])
                blueC += 1
                
        self.graphics.drawState(self.redTeam, self.blueTeam)

    def gameLoop(self):
        elapsedTurns = 0
        
        initialRedStr, initialBlueStr = utilities.get_team_strengths(self.redTeam, self.blueTeam)
        
        while(((len(self.redTeam.armies) > 0) and (len(self.blueTeam.armies) > 0)) and elapsedTurns < 10):
            print("\n\n=====TURN {}=====".format(elapsedTurns))
                        
            packet = self.createDataPacket() #Minmax algorithm uses relative direction so need to use a packet that has relative direction
            utilities.printPacket(packet)
            
            if packet[0][0] == 0:
                controller = self.redController
            elif packet[0][0] == 1:
                controller = self.blueController
                
            #Minmax
            if controller == "minmax":
                AIMove = self.minmaxAI.get_move(packet)[1:]
            #dumbAI
            elif controller == "dumbai":
                AIMove = self.dumbAI.get_move(packet)
            #random ai
            elif controller == "randai":
                AIMove = self.randAI.get_move()
            elif controller == "maxnet":
                AIMove = self.maxnetAI.get_move(packet)
            #Neural network
            elif controller == "neurnet":
                if self.nnet_train:
                    mmMove = self.minmaxAI.get_move(packet, 1)[1:]
                    #When the armies are very close or very far minmax tends to not move.
                    #dumbAI is used to give some variety when this happens
                    if mmMove[0] == 0:
                        chance = random.randint(0,2)
                        if chance != 0 or mmMove[1] == 0:
                            mmMove = self.dumbAI.get_move(packet)
                    print(mmMove)
                    self.neurnetAI.learning_step(packet, mmMove)
                    AIMove = mmMove
                else:
                    AIMove = self.neurnetAI.get_move(packet)

            print(AIMove)
            magnitude = AIMove[0]
            direction = AIMove[1]

            self.turnOrder[0].move(magnitude, direction)

            isDead = self.calculateCombat(self.turnOrder[0])
            if not isDead:
                self.prepNextTurn()
            self.graphics.drawState(self.redTeam, self.blueTeam)
            elapsedTurns += 1

            #Check if end state
            rStr, bStr = utilities.get_team_strengths(self.redTeam, self.blueTeam)
            if self.checkEndState():
                return [[initialRedStr, initialBlueStr], [rStr, bStr]]

        #Return results if out of loop
        rStr, bStr = utilities.get_team_strengths(self.redTeam, self.blueTeam)
        return [[initialRedStr, initialBlueStr], [rStr, bStr]]
    
    def checkEndState(self):
        rStr, bStr = utilities.get_team_strengths(self.redTeam, self.blueTeam)
        if rStr <= 0 or bStr <= 0:
            return True
        else:
            return False


    def calculateCombat(self, unit):
        
        def calculateDamage(attDirection, attStr, defStr, Bonus, defOutOfRange):
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

        def check_Attacker_Bonus_Range(unit1, unit2):
            #Check if unit1 is attacking unit2
            if utilities.get_relative_direction(unit1, unit2) < utilities.get_relative_direction(unit2, unit1):
                Attacker = True
            else:
                Attacker = False

            dist = utilities.get_distance(unit1, unit2)
            if unit1.fireRange > dist and unit2.fireRange > dist:
                #They're both in range of each other
                Range = False 
            else:
                Range = True
                if unit1.fireRange > unit2.fireRange:
                    Attacker = True
                else:
                    Attacker = False
                    
            if not Attacker and utilities.get_relative_direction(unit1, unit2) > 30 and utilities.get_relative_direction(unit2, unit1) < 90:
                Bonus = True
            elif Attacker and utilities.get_relative_direction(unit2, unit1) > 30 and utilities.get_relative_direction(unit1, unit2) < 90:
                Bonus = True
            else:
                Bonus = False
                
            return Attacker, Bonus, Range

        for i in range(len(self.turnOrder)):
            if unit == self.turnOrder[i] or unit.get_unitTeam() == self.turnOrder[i].get_unitTeam():
                continue
            if self.turnOrder[i].strength <= 0:
                continue
            if unit.strength <= 0:
                break
            dist = utilities.get_distance(unit, self.turnOrder[i])
            if (dist < unit.fireRange or dist < self.turnOrder[i].fireRange):
                Attacker, Bonus, Range = check_Attacker_Bonus_Range(unit, self.turnOrder[i])
                if Attacker:
                    attDirection = utilities.get_relative_direction(self.turnOrder[i], unit)
                    dmgDone, dmgTaken = calculateDamage(attDirection,
                                                        unit.strength, self.turnOrder[i].strength,
                                                        Bonus, Range)
                else:
                    attDirection = utilities.get_relative_direction(unit, self.turnOrder[i])
                    dmgTaken, dmgDone = calculateDamage(attDirection,
                                                        self.turnOrder[i].strength, unit.strength,
                                                        Bonus, Range)
                unit.strength -= dmgTaken
                self.turnOrder[i].strength -= dmgDone
                
        unitIsDead = False #The unit that is taking its turn
        for army in self.turnOrder:
            if army.strength <= 0:
                if army == self.turnOrder[0]:
                    unitIsDead = True
                army.strength = 0
                self.turnOrder.remove(army)
        return unitIsDead
                

    def createDataPacket(self):
        #unit is the currently selected unit and is the focus of the packet
        unit = self.turnOrder[0]
        dataPacket = [unit.get_packet() + [0, 0]]
        i = 1
        while len(dataPacket) < len(self.turnOrder):
            temp = self.turnOrder[i].get_packet()
            temp += [utilities.get_distance(unit, self.turnOrder[i]),
                     utilities.get_absolute_direction(unit, self.turnOrder[i])]
            dataPacket.append(temp)
            i += 1
            
        return dataPacket
    
    def prepNextTurn(self):
        self.turnOrder.append(self.turnOrder[0])
        self.turnOrder.remove(self.turnOrder[0])
    
    def save_model(self):
        self.Brain.save_model()
        
    def doOneTurn(self):
        packet = self.createDataPacket() #Minmax algorithm uses relative direction so need to use a packet that has relative direction
        
        if packet[0][0] == 0:
            controller = self.redController
        elif packet[0][0] == 1:
            controller = self.blueController
            
        #Minmax
        if controller == "minmax":
            AIMove = self.minmaxAI.get_move(packet)[1:]
        #dumbAI
        elif controller == "dumbai":
            AIMove = self.dumbAI.get_move(packet)
        #random ai
        elif controller == "randai":
            AIMove = self.randAI.get_move()
        #Neural network
        elif controller == "neurnet":
            if self.nnet_train:
                mmMove = self.minmaxAI.get_move(packet, 1)[1:]
                #When the armies are very close or very far minmax tends to not move.
                #dumbAI is used to give some variety when this happens
                if mmMove[0] == 0:
                    chance = random.randint(0,2)
                    if chance != 0 or mmMove[1] == 0:
                        mmMove = self.dumbAI.get_move(packet)
                print(mmMove)
                self.neurnetAI.learning_step(packet, mmMove)
                AIMove = mmMove
            else:
                AIMove = self.neurnetAI.nnet_move(packet)

        magnitude = AIMove[0]
        direction = AIMove[1]

        self.turnOrder[0].move(magnitude, direction)
        
        isDead = self.calculateCombat(self.turnOrder[0])
        if not isDead:
            self.prepNextTurn()
        
        return [packet, AIMove]
        
