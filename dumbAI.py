# PACKET LAYOUT
# [Team, Str, fireRange, moveSpeed, x, y, heading, distance, direction]
#   0     1       2         3       4  5     6        7          8


class dumbAI(object):
    def __init__(self):
        pass

    def get_move(self, packet):

        i = 1
        #Find next army on a different team
        while packet[i][0] == packet[0][0]:
            i += 1
        nextOpponent = packet[i]
        direction = packet[i][8]
        distance = nextOpponent[7]-10 #10 unit buffer
        for i in range(11):
            newDist = (i/10) * packet[0][3]
            if newDist > distance:
                if i > 0:
                    i -= 1
                break
        return [i, direction]
