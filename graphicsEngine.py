import turtle

class graphicsEngine(object):
    def __init__(self):
        self.Ramses = turtle
        self.Ramses.begin_fill()
        self.Ramses.speed(0)
        self.Ramses.mode('logo')

    def drawState(self, red, blue):
        self.Ramses.clearscreen()
        self.Ramses.hideturtle()
        self.Ramses.penup()
        self.Ramses.speed(0)
        for unit in red.get_armies():
            if unit.strength <= 0:
                continue
            self.Ramses.color("red")
            self.Ramses.goto(unit.location)
            self.Ramses.setheading(unit.heading)
            self.Ramses.stamp()
            self.Ramses.write("{} | {} | {}".format(unit.get_unitID(), unit.strength, unit.heading))
            self.drawRangeCircle(unit.fireRange)
            
        for unit in blue.get_armies():
            if unit.strength <= 0:
                continue
            self.Ramses.color("blue")
            self.Ramses.goto(unit.location)
            self.Ramses.setheading(unit.heading)
            self.Ramses.stamp()
            self.Ramses.write("{} | {} | {}".format(unit.get_unitID(), unit.strength, unit.heading))
            self.drawRangeCircle(unit.fireRange)
            
        self.Ramses.hideturtle()

    def drawRangeCircle(self, fireRange):
        curPos = self.Ramses.position()
        self.Ramses.setheading(0)
        newX = curPos[0] + fireRange
        self.Ramses.setx(newX)
        self.Ramses.pendown()
        self.Ramses.circle(fireRange)
        self.Ramses.penup()
