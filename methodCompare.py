import pandas as pd
import random

from Engine import Engine

columns = ["Num_Red_Armies", "Num_Blue_Armies", "Red_Controller", "Init_Red_Str",
           "Final_Red_Str", "Blue_Controller", "Init_Blue_Str", "Final_Blue_Str",
           "Victor"]

#redSize, blueSize, redController, InitRed, FinalRed, blueController, InitBlue, FinalBlue, Victor
#   0        1            2           3         4            5           6         7         8


def evalMethod(results, controllerToEvaluate = "maxnet"):
    games = 0
    wins = 0
    favoredGames = 0
    favoredWins = 0
    equalGames = 0
    equalWins = 0
    underdogGames = 0
    underdogWins = 0
    for i in range(len(results)):
        games += 1
        if results[i][2] == controllerToEvaluate:
            controllerTeam = 'r'
        else:
            controllerTeam = 'b'
            
        if results[i][3] > results[i][6]:
            favored = 'r'
        elif results[i][3] == results[i][6]:
            favored = 'n'
        else:
            favored = 'b'
        
        if controllerTeam == results[i][8]:
            controllerWin = True
        else:
            controllerWin = False
        
        if controllerWin:
            wins += 1
            
        if controllerTeam == favored:
            favoredGames += 1
            if controllerWin:
                favoredWins += 1
        elif favored == 'n':
            equalGames += 1
            if controllerWin:
                equalWins += 1
        else:
            underdogGames += 1
            if controllerWin:
                underdogWins += 1
                
    print("\nWinrate:", (wins/games), "\nWins, Games:", wins, games)
    print("\nFavored Winrate:", (favoredWins/favoredGames), "\nWins, Games:", favoredWins, favoredGames)
    print("\nEquivalent Winrate:", (equalWins/equalGames), "\nWins, Games:", equalWins, equalGames)
    print("\nUnderdog Winrate:", (underdogWins/underdogGames), "\nWins, Games:", underdogWins, underdogGames)


def compareMethod(redController, blueController, fileName):
    data = []
    c_eval = blueController
    sizeArray = [random.randint(1,2), random.randint(1,2)]
    controllers = [redController, blueController]
    engine = Engine(sizeArray, controllers, nnet_train = False)
    for i in range(100):
        print("GAME:", i+1)
        try:
            results = engine.gameLoop()
            container = [sizeArray[0], sizeArray[1], controllers[0], results[0][0], results[1][0],
                         controllers[1], results[0][1], results[1][1]]
            
            if results[1][0] <= 0 and results[1][1] <= 0:
                victor = "t"
            elif container[3]-container[4] > container[6]-container[7]:
                victor = "b"
            elif container[3]-container[4] < container[6]-container[7]:
                victor = "r"
            else:
                victor = "t"
    
            container.append(victor)
            data.append(container)
            c1 = random.randint(0,1)
            c2 = c1 * -1 + 1
            controllers = [controllers[c1], controllers[c2]]
            
            sizeArray = [random.randint(1,2), random.randint(1,2)]
            engine.reset(sizeArray, controllers, allowRandom = True)
            engine.sarsa.freeSpace(5)
        except KeyboardInterrupt:
            break
        
    df = pd.DataFrame(data = data)
    df.to_csv("{}.csv".format(fileName), index = False, header = columns)
    
    evalMethod(data, c_eval)

if __name__ == "__main__":
    compareMethod("dumbai", "sarsa", "dumb_v_sarsa")
