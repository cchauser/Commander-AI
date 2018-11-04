# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:52:06 2018

@author: Cullen
"""

from Engine import Engine
from random import randint

"""
Packet structure:
    [Team, Strength, Fire range, Movement Speed, x-coordinate, y-coordinate, Heading, Distance, Direction]
    
Turn & Movement:
    At the start of each turn the engine will print the current game state in the packet structure above, one unit per row.
    Distance and Direction are relative to the first unit in the packet.

    The engine will then ask the player that controls the first unit how they would like to move their unit.
    The input should be in the form "magnitude direction". Magnitude should be in the range [0,10], direction should be [0,359].
    Some edge cases for inputs are currently handled but not all. The AIs will never abuse edge cases.
    
    Units move according to this formula:
        delta = movementSpeed/magnitude
        

Combat:
    A unit is considered to be the attacker if their heading is closer to the direction of their opponent than their opponent's
    heading to that unit's direction.
    if (|unit1Heading - [direction of unit2 from unit1]| < |unit2Heading - [direction of unit1 from unit2]|):
        unit1 is attacking.
    
    Flanking:
        Units will get a bonus to damage dealt and reduced damage taken if flanking.
        A unit is considered to be flanking if all of these are true:
            * Is the attacker
            * Attacker's direction from defender's current heading is > 30°
                e.g. I'm facing east, my opponent is attacking from the north. The attacker's relative direction from my heading is 90°
            * Defender's direction from attacker's current heading is < 90°
            
        Flanks do more damage the closer the attacker's heading is to the direction of the defender.

AI:
    Random AI:
        Controller: 'randai'
        Difficulty: You'll win everytime
        
        Moves randomly and without purpose.
        
    Dumb AI:
        Controller: 'dumbai'
        Difficulty: Moderate
        
        Charges a random opponent. Will never move closer than 20 units
        
    Minmax AI:
        Controller: 'minmax'
        Difficulty: Hard
        
        Plays in a way that maximizes its own chances of winning. Will likely abuse flanking. Only looks two moves ahead.
        
    Neural Network AI:
        Controller: 'neurnet'
        Difficulty: Moderate-Hard
        
        Moves similarly to the minmax AI but doesn't have the accuracy to abuse flanking.
        
    Maxnet AI:
        Controller: 'maxnet'
        Difficulty: Very Hard
        
        The neural network augments the minmax algorithm, allowing the minmax algorithm to look further ahead without compromising speed.
        Likely to abuse flanking. Looks ahead 3 moves.
"""


controller1 = 'human'
controller2 = 'dumbai'

armySizes = [randint(1,2), randint(1,2)]
controllers = [controller1, controller2]

engine = Engine(armySizes, controllers)
results = engine.gameLoop()

if results[1][0] <= 0 and results[1][1] <= 0:
    print("It's a tie!")
elif results[0][0]-results[1][0] > results[0][1]-results[1][1]:
    print("{} wins".format(controllers[1]))
elif results[0][0]-results[1][0] < results[0][1]-results[1][1]:
    print("{} wins".format(controllers[0]))
else:
    print("It's a tie!")


























