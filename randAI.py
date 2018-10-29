import random

class randAI(object):
    def __init__(self):
        pass

    def get_move(self):
        magnitude = random.randint(0,10)
        direction = random.randint(0,359)

        return [magnitude, direction]
