import random
from AI import AI

class randAI(AI):
    def __init__(self):
        pass

    def get_move(self, _unused1, _unused2):
        magnitude = random.randint(0,10)
        direction = random.randint(0,359)

        return [magnitude, direction]
