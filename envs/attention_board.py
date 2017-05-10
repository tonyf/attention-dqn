from PIL import Image
import numpy as np
import random

DOT=225
WALLS=127
AGENT=85

first = True

MAX_MOVE=8
MOVES = {   0: [ 0,  0],
            1: [-1,  0],
            2: [ 1,  0],
            3: [-1,  0],
            4: [ 1,  0],
            5: [-1, -1],
            6: [ 1, -1],
            7: [-1,  1],
            8: [ 1,  1],
        }

class AttentionBoard(object):
    def __init__(self, size, radius=2, timestep=.25, speed=1.5):
        self.board = np.zeros((size, size), dtype=float)
        self.size = size
        self.radius = radius
        self.timestep = timestep
        self.time = 0.0
        self.speed = speed

        self.dot = np.array([size/2, size/2], dtype=float)
        self.dot_v = np.zeros((2))
        self.dot_a = np.zeros((2))

        self.agent = self.constrain_pos(np.random.randint(self.size, size=2))
        self.update_board()

    def step(self, action):
        move = int(action[0])
        if move < 0 or move > MAX_MOVE:
            return -1.
        self.agent = self.constrain_pos(self.agent + self.speed*np.asarray(MOVES[move]))
        self.update_board()
        return self.reward(mode='distance')
        

    """ Get reward for being attentive to a certain pixel """
    def reward(self, scale=1, mode='binary'):
        # if np.array_equal(self.agent, self.dot):
        if self.does_overlap(self.agent, self.dot):
            return 1.0 * scale
        if mode == 'binary':
            return 0.0
        if mode == 'distance':
            return -1 * np.linalg.norm(self.dot - self.agent)


    """ Get image from board """
    def image(self):
        return self.board

    """ Update the board  """
    def next(self, acceleration=None, mode='random'):
        if mode == 'static':
            done = np.array_equal(self.agent, self.dot)
            return (self.image(), done)
        if acceleration:
            return (self._next(acceleration), False)
        return (self._rand_next(), False)


    """ Helper Functions """

    def does_overlap(self, agent, dot):
        agent_set = set(self.circle_points(agent, self.radius))
        dot_set = set(self.circle_points(dot, self.radius))
        intersection = list(agent_set & dot_set)
        if len(intersection) > 0:
            return True
        return False

    def calculate_pos(self, pos, v, a, t):
        pos = pos + (v * t) + (0.5 * a * t * t)
        return self.constrain_pos(pos)
    
    def constrain_pos(self, pos):
        if pos[0] - self.radius <= 0:
            pos[0] = 0 + self.radius
            self.dot_v[0] = 0
        if pos[1] - self.radius <= 0:
            pos[1] = 0 + self.radius
            self.dot_v[1] = 0
        if pos[0] + self.radius >= self.size:
            pos[0] = self.size - self.radius - 1
            self.dot_v[0] = 0
        if pos[1] + self.radius >= self.size:
            pos[1] = self.size - self.radius - 1
            self.dot_v[1] = 0
        return pos
    
    def circle_points(self, point, radius):
        x_center, y_center = point

        x_center = int(x_center)
        y_center = int(y_center)
        radius = int(radius)
        r_squared = radius*radius

        circle_points = []
        for x in xrange(x_center-radius, x_center+radius):
            for y in xrange(y_center-radius, y_center+radius):
                if (x-x_center)*(x-x_center) + (y-y_center)*(y_center) <= r_squared:
                    x_sym = x_center - (x - x_center)
                    y_sym = y_center - (y - y_center)
                    circle_points.append((x,y))
                    circle_points.append((x,y_sym))
                    circle_points.append((x_sym,y))
                    circle_points.append((x_sym,y_sym))
        return circle_points
    
    def write_circle(self, point, radius, color):
        circle_points = self.circle_points(point, radius)
        
        for point in circle_points:
            x, y = point
            self.board[x,y] = color

    def update_board(self):
        # Clear the board
        self.board.fill(0)
        # Write position of dot
        self.write_circle(self.dot, self.radius, DOT)
        # Write position of agent
        self.write_circle(self.agent, self.radius, AGENT)

    def _rand_next(self):
        a = np.random.uniform(low=-1.0, high=1.0, size=2)
        return self._next(a)

    def _next(self, a):
        self.time += self.timestep
        self.dot_a = a
        self.dot_v = self.dot_v + self.dot_a * self.timestep
        self.dot = self.calculate_pos(self.dot, self.dot_v, self.dot_a, self.timestep)

        self.update_board()
        return self.image()