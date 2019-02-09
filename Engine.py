import numpy as np
import random

from Team import Team
from graphicsEngine import graphicsEngine
from minmaxAI import minmaxAI
from dumbAI import dumbAI
from randAI import randAI
from neurnetAI import neurnetAI
from maxnetAI import maxnetAI
from SARSA import SARSA
from Utility import Utility

utilities = Utility()

class Engine(object):
    def __init__(self, armySizeArray, controllerArray, nnet_train = False):
        self.graphics = graphicsEngine()
        armySizeArray = np.asarray(armySizeArray)
        self.Controllers = controllerArray
        self.nnet_train = nnet_train
        self.t = 1 # This is for the neural network training
        self.teamArray = []
        
        totalSoldiers = 0
        for i in range(len(armySizeArray)):
            spawn = [20 * (len(armySizeArray) - 2) * (-1)**i, 20 * (len(armySizeArray) -1) * (-1)**i]
            self.teamArray.append(Team(i, armySizeArray[i], spawn, totalSoldiers))
            totalSoldiers += armySizeArray[i]

        self.turnOrder = []
        totalArmies = np.sum(armySizeArray)
        armySizeArray -= 1 #Set up for pulling armies out of team arrays
        while len(self.turnOrder) < totalArmies:
            for i in range(len(self.teamArray)):
                if armySizeArray[i] < 0:
                    continue
                self.turnOrder.append(self.teamArray[i][armySizeArray[i]])
                armySizeArray[i] -= 1

        self.adjustHeadingsForStartOfGame()
        self.graphics.drawState(self.teamArray[0], self.teamArray[1]) #TODO: more than 2 team compatibility

    def reset(self, armySizeArray, controllerArray, allowRandom = False):
        for controller in self.Controllers:
            controller.free_space()
        
        if allowRandom:
            randomSpawn = random.randint(0,1) #Random spawns occur when value != 0
        else:
            randomSpawn = 0
            
        armySizeArray = np.asarray(armySizeArray)
        self.Controllers = controllerArray
        self.teamArray = []
        
        totalSoldiers = 0
        for i in range(len(armySizeArray)):
            spawn = [20 * (len(armySizeArray) - 2) * (-1)**i, 20 * (len(armySizeArray) -1) * (-1)**i]
            self.teamArray.append(Team(i, armySizeArray[i], spawn, totalSoldiers, randomSpawn))
            totalSoldiers += armySizeArray[i]

        self.turnOrder = []
        totalArmies = np.sum(armySizeArray)
        armySizeArray -= 1 #Set up for pulling armies out of team arrays
        while len(self.turnOrder) < totalArmies:
            for i in range(len(self.teamArray)):
                if armySizeArray[i] < 0:
                    continue
                self.turnOrder.append(self.teamArray[i][armySizeArray[i]])
                armySizeArray[i] -= 1
                
        self.adjustHeadingsForStartOfGame()
        self.graphics.drawState(self.teamArray[0], self.teamArray[1]) #TODO: more than 2 team compatibility

    def gameLoop(self):
        elapsedTurns = 0
        
        initialStrength = utilities.get_team_strengths(self.teamArray)
        
        while((not self.checkEndState()) and elapsedTurns < 10):
            print("\n\n=====TURN {}=====".format(elapsedTurns))
                        
            packet = self.createDataPacket() #Minmax algorithm uses relative direction so need to use a packet that has relative direction
            utilities.printPacket(packet)
            
            controller = self.Controllers[packet[0][0]]
                
            Move = controller.get_move(packet)

            print(Move)
            magnitude = Move[0]
            direction = Move[1]

            self.turnOrder[0].move(magnitude, direction)

            isDead = self.calculateCombat(self.turnOrder[0])
            if not isDead:
                self.prepNextTurn()
            self.graphics.drawState(self.teamArray[0], self.teamArray[1]) #TODO: more than 2 team compatibility
            elapsedTurns += 1


        #Return results of battle
        endStrength = utilities.get_team_strengths(self.teamArray)
        return [initialStrength, endStrength]
    
    def checkEndState(self):
        strengths = utilities.get_team_strengths(self.teamArray)
        numCombatantTeams = 0
        for s in strengths:
            if s > 0:
                numCombatantTeams += 1
            if numCombatantTeams > 1:
                return False
        return True


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
                Range = True #One outranges the other
                #Attacker is the one that outranges
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
    
    #Sets the unit's headings to the average direction of the opponents.
    def adjustHeadingsForStartOfGame(self):
        for unit in self.turnOrder:
            headingArray = []
            for otherUnit in self.turnOrder:
                if otherUnit == unit:
                    continue
                elif otherUnit.get_unitTeam() == unit.get_unitTeam():
                    continue
                else:
                    headingArray.append(utilities.get_absolute_direction(unit, otherUnit))
            headingArray = np.asarray(headingArray)
            if len(headingArray) > 1:
                if np.max(headingArray) - np.min(headingArray) > 180:
                    newHeading = (np.max(headingArray) + (((headingArray[1] - headingArray[0]) + 180) % 360 - 180)/2) + random.randint(-15,15)
                else:
                    newHeading = (np.max(headingArray) - (((headingArray[1] - headingArray[0]) + 180) % 360 - 180)/2) + random.randint(-15,15)
            else:
                newHeading = headingArray[0] + random.randint(-15,15)
            unit.set_heading(int(newHeading))
                    
    def doOneTurn(self):
        packet = self.createDataPacket() #Minmax algorithm uses relative direction so need to use a packet that has relative direction
        
        controller = self.Controllers[packet[0][0]]
            
        #Minmax
        if controller == "minmax":
            AIMove = self.minmaxAI.get_move(packet)[1:]
        #dumbAI
        elif controller == "dumbai":
            AIMove = self.dumbAI.get_move(packet)
        #random ai
        elif controller == "randai":
            AIMove = self.randAI.get_move()
        #Maxnet AI
        elif controller == "maxnet":
            AIMove = self.maxnetAI.get_move(packet)
        #Neural network
        elif controller == "neurnet":
            AIMove = self.neurnetAI.get_move(packet)

        magnitude = AIMove[0]
        direction = AIMove[1]

        self.turnOrder[0].move(magnitude, direction)
        
        isDead = self.calculateCombat(self.turnOrder[0])
        if not isDead:
            self.prepNextTurn()
        
        return [packet, AIMove]
        
