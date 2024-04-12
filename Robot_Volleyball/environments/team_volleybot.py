import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import cv2  # installed with gym anyways
# from collections import deque
from time import sleep
import copy

# 引入webot模块
try:
    from controller import Supervisor, Keyboard

    game_keyboard = Keyboard()
    is_first_key_init = True

    WEBOTS_MODE = True
except:
    WEBOTS_MODE = False

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)

RENDER_MODE = True
TIMESTEP = 1 / 30.
NUDGE = 0.1
FRICTION = 1.0
INIT_DELAY_FRAMES = 30
GRAVITY = -9.8 * 2 * 1.5
MAXLIVES = 5

VOLLEYBALL_IDENTITY = ["Hitter", "Libero", "Utility", "Hitter", "Libero"]  # 一个角色列表，不同数目的球员时从此队列中依次取得身份
"""may还可以设置一下不同位置
Hitter：进攻者（好几种细分，不同位置？）
Libero：接应，防守和接发球
Utility：自由人（第三个其实有点纠结...先设置为自由人吧
"""
IDENTITY_REWARD_WEIGHT = {
    "Hitter": {
        "ball_bonus": 2,
        "bonus": 1,  # 其实是side_bonus，因历史原因就没改了
        "malus": 0.5,  # 其实是碰撞惩罚
        "result_fail": 0.5,
        "result_win": 2,
        "dis_malus": 2
    },
    "Libero": {
        "ball_bonus": 2,
        "bonus": 1,  # 其实是side_bonus，因历史原因就没改了
        "malus": 2,  # 其实是碰撞惩罚
        "result_fail": 2,
        "result_win": 0.5,
        "dis_malus": 1
    },
    "Utility": {
        "ball_bonus": 1,
        "bonus": 1,  # 其实是side_bonus，因历史原因就没改了
        "malus": 1,  # 其实是碰撞惩罚
        "result_fail": 1,
        "result_win": 1,
        "dis_malus": 0.5
    },
}
IF_MULTI_ROLE = True  # 角色分工；为False时则为位置分工

### Webots settings:
TIME_STEP = 32
if WEBOTS_MODE:
    supervisor = Supervisor()
    game_keyboard.enable(TIME_STEP)  # 依其进行频率初始化

left_controlled_list = [False for _ in range(3)]
right_controlled_list = [False for _ in range(3)]
CUR_KEY = 0


class World:
    def __init__(self, update=True):

        self.width = 24 * 2
        self.height = self.width
        self.max_depth = self.width / 2
        self.step = -np.inf
        self.depth = -np.inf
        self.wall_width = 1.0
        self.wall_height = 2.0
        self.wall_depth = -np.inf
        self.player_vx = 10 * 1.75
        self.player_vy = 10 * 1.35
        self.player_vz = 10 * 1.75
        self.max_ball_v = 15 * 1.5
        self.gravity = -9.8 * 2 * 1.5
        self.update = update
        self.stuck = False
        self.setup()

    def setup(self, n_update=4, init_depth=6):
        if not self.update:
            self.wall_depth = self.depth = self.max_depth
        elif self.stuck:
            self.step = 0
            self.wall_depth = self.depth = init_depth
        else:
            self.step = self.max_depth / n_update
            self.wall_depth = self.depth = init_depth

    def update_world(self):
        if self.update:
            self.depth += self.step
            self.wall_depth = self.depth

        if self.depth >= self.max_depth:
            self.depth = self.max_depth
            self.wall_depth = self.depth
            self.update = False


WORLD = World()


class DelayScreen:
    def __init__(self, life=INIT_DELAY_FRAMES):
        self.life = 0
        self.reset(life)

    def reset(self, life=INIT_DELAY_FRAMES):
        self.life = life

    def status(self):
        if (self.life == 0):
            return True
        self.life -= 1
        return False


class Particle:
    def __init__(self, x, y, z, vx, vy, vz, r, name):
        self.x = x
        self.y = y
        self.z = z
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.r = r
        if WEBOTS_MODE:
            self.particule = supervisor.getFromDef(name)
            self.location = self.particule.getField("translation")
            self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

    def move(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP
        self.z += self.vz * TIMESTEP
        if WEBOTS_MODE:
            self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

    def applyAcceleration(self, ax, ay, az):
        self.vx += ax * TIMESTEP
        self.vy += ay * TIMESTEP
        self.vz += az * TIMESTEP * (
                WORLD.depth / WORLD.max_depth)  # Keep the z-axis proportional to the actual depth

    def checkEdges(self):
        if (self.x <= (self.r - WORLD.width / 2)):
            self.vx *= -FRICTION
            self.x = self.r - WORLD.width / 2 + NUDGE * TIMESTEP
        if (self.x >= (WORLD.width / 2 - self.r)):
            self.vx *= -FRICTION;
            self.x = WORLD.width / 2 - self.r - NUDGE * TIMESTEP

        if WORLD.depth >= self.r:
            if (self.z <= (self.r - WORLD.depth / 2)):  # 中间是0，depth左右两边是正负
                self.vz *= -FRICTION * (WORLD.depth / WORLD.max_depth)
                self.z = self.r - WORLD.depth / 2 + NUDGE * TIMESTEP
            if (self.z >= (WORLD.depth / 2 - self.r)):
                self.vz *= -FRICTION * (WORLD.depth / WORLD.max_depth);
                self.z = WORLD.depth / 2 - self.r - NUDGE * TIMESTEP
        else:
            if (self.z <= (WORLD.depth / 2)):
                self.vz *= -FRICTION * (WORLD.depth / WORLD.max_depth)
                self.z = -WORLD.depth / 2 + NUDGE * TIMESTEP

            if (self.z >= (WORLD.depth / 2)):
                self.vz *= -FRICTION * (WORLD.depth / WORLD.max_depth);
                self.z = WORLD.depth / 2 - NUDGE * TIMESTEP

        # 如果particle碰到地板，防止其穿过地面
        if (self.y <= (self.r)):
            self.vy *= -FRICTION
            self.y = self.r + NUDGE * TIMESTEP
            if WEBOTS_MODE:
                self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

            if (self.x <= 0):
                return -1  # The left player loses a life
            else:
                return 1  # The right player loses a life

        # 避免超越世界的高度
        if (self.y >= (WORLD.height - self.r)):
            self.vy *= -FRICTION
            self.y = WORLD.height - self.r - NUDGE * TIMESTEP

        # 避免particle越过球网
        if ((self.x <= (WORLD.wall_width / 2 + self.r)) and (self.prev_x > (WORLD.wall_width / 2 + self.r)) and (
                self.y <= WORLD.wall_height)):
            self.vx *= -FRICTION
            self.x = WORLD.wall_width / 2 + self.r + NUDGE * TIMESTEP

        if ((self.x >= (-WORLD.wall_width / 2 - self.r)) and (self.prev_x < (-WORLD.wall_width / 2 - self.r)) and (
                self.y <= WORLD.wall_height)):
            self.vx *= -FRICTION
            self.x = -WORLD.wall_width / 2 - self.r - NUDGE * TIMESTEP

        if WEBOTS_MODE:
            self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

        return 0;

    def getDist2(self, p):
        dz = p.z - self.z
        dy = p.y - self.y
        dx = p.x - self.x
        return (dx * dx + dy * dy + dz * dz)

    def isColliding(self, p):
        r = self.r + p.r
        if WORLD.depth != 0:
            # if distance is less than total radius and the depth, then colliding.
            return (r * r > self.getDist2(p) and (self.z * self.z <= WORLD.wall_depth * WORLD.wall_depth))
        else:
            return (r * r > self.getDist2(p))

    def bounce(self, p):
        abx = self.x - p.x
        aby = self.y - p.y
        abz = self.z - p.z
        abd = math.sqrt(abx * abx + aby * aby + abz * abz)
        abx /= abd
        aby /= abd
        abz /= abd
        nx = abx
        ny = aby
        nz = abz
        abx *= NUDGE
        aby *= NUDGE
        abz *= NUDGE

        while (self.isColliding(p)):
            self.x += abx
            self.y += aby
            self.z += abz
            if WEBOTS_MODE:
                self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

        ux = self.vx - p.vx
        uy = self.vy - p.vy
        uz = self.vz - p.vz
        un = ux * nx + uy * ny + uz * nz
        unx = nx * (un * 2.)
        uny = ny * (un * 2.)
        unz = nz * (un * 2.)
        ux -= unx
        uy -= uny
        uz -= unz
        self.vx = ux + p.vx
        self.vy = uy + p.vy
        self.vz = (uz + p.vz)

    def limitSpeed(self, minSpeed, maxSpeed):
        mag2 = self.vx * self.vx + self.vy * self.vy + self.vz * self.vz;
        if (mag2 > (maxSpeed * maxSpeed)):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vz /= mag
            self.vx *= maxSpeed
            self.vy *= maxSpeed
            self.vz *= maxSpeed * (
                    WORLD.depth / WORLD.max_depth)
        if (mag2 < (minSpeed * minSpeed)):
            mag = math.sqrt(mag2)
            self.vx /= mag
            self.vy /= mag
            self.vz /= mag
            self.vx *= minSpeed
            self.vy *= minSpeed
            self.vz *= minSpeed * (
                    WORLD.depth / WORLD.max_depth)


class Team():

    def __init__(self, dir, name, n_mates=3, diff_identity=True):
        self.name = name
        self.n_mates = n_mates
        self.team = {}
        self.dir = dir
        self.setTeam()
        self.life = MAXLIVES

    def lives(self):

        return self.life

    def setTeam(self):
        for i in range(self.n_mates):
            self.team[self.name + str(i + 1)] = Agent(self.dir, (self.dir * 9) + (self.dir * i * 6), 0,
                                                      0,
                                                      self.name + str(i + 1),
                                                      identity=VOLLEYBALL_IDENTITY[i]
                                                      )
            """self.team[self.name+str(i+1)] = Agent(self.dir, (self.dir*12), 0, 
                                                 self.dir*4 - self.dir * i*8, 
                                                 self.name+str(i+1)
                                                 )"""

        for agt in self.team:
            self.team[agt].getMates(self.team)

    def setAction(self, actions: list):

        for i in range(len(actions)):
            agent = self.team[self.name + str(i + 1)]
            if (not agent.controlled_by_board) or (not WEBOTS_MODE):
                agent.setAction(list(actions[i]))
            else:
                # key = game_keyboard.getKey()
                global CUR_KEY
                action = [0 for _ in range(5)]
                # 先只能同时输入一个吧
                if CUR_KEY == Keyboard.UP:
                    # 执行向上操作
                    action[0] = 1
                    pass
                elif CUR_KEY == Keyboard.DOWN:
                    # 执行向下操作
                    action[1] = 1
                    pass
                elif CUR_KEY == Keyboard.LEFT:
                    # 执行向左操作
                    if agent.dir == -1:  # 进行协调
                        action[4] = 1
                    else:
                        action[3] = 1
                    pass
                elif CUR_KEY == Keyboard.RIGHT:
                    if agent.dir == -1:
                        action[3] = 1
                    else:
                        action[4] = 1
                    # 执行向右操作
                    pass
                elif CUR_KEY == ord(' '):
                    action[2] = 1
                    # 空格操作
                    # 在这里编写你的代码
                    pass
                agent.setAction(list(action))

    def teamMove(self):
        for i in range(self.n_mates):
            agent = self.team[self.name + str(i + 1)]
            agent.move()

    def update(self):
        for i in range(self.n_mates):
            agent = self.team[self.name + str(i + 1)]
            agent.update()

    def getObservations(self, ball, opponents):
        obs = []
        for i in range(self.n_mates):
            agent = self.team[self.name + str(i + 1)]
            obs.append(agent.getObs(ball, opponents))
        return [ob.reshape(1, -1) for ob in obs]
        # return obs

    def getTeamState(self):
        states = []
        for i in range(self.n_mates):
            agent = self.team[self.name + str(i + 1)]
            states += [agent.x, agent.y, agent.z, agent.vx, agent.vy, agent.vz]

        return states


class Agent():
    def __init__(self, dir, x, y, z, name, identity="", controlled_by_board: bool = False):

        self.dir = dir  # -1 means left, 1 means right player for symmetry.
        self.x = x
        self.y = y
        self.z = z

        # self.r = 1.5
        self.r = 3
        self.name = name
        self.side = int(name[-1])
        self.ball_bonus = 0
        self.dis_malus = 0
        self.bonus = 0
        self.malus = 0

        self.BallCollisionS = 0

        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.desired_vx = 0
        self.desired_vy = 0
        self.desired_vz = 0
        self.emotion = "happy"
        self.life = MAXLIVES
        self.mates = {}
        if WEBOTS_MODE:
            self.agent = supervisor.getFromDef(name)
            self.location = self.agent.getField("translation")
            self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

        if identity != "":
            self.identity = identity

        self.controlled_by_board = controlled_by_board

    def lives(self):
        return self.life

    def setAction(self, action):
        action = list(action)
        # print(action)
        forward = False
        backward = False
        jump = False
        right = False
        left = False

        if action[0] > 0:
            forward = True
        if action[1] > 0:
            backward = True
        if action[2] > 0:
            jump = True
        if action[3] > 0:
            right = True
        if action[4] > 0:
            left = True

        self.desired_vx = 0
        self.desired_vy = 0
        self.desired_vz = 0

        if (forward and (not backward)):
            self.desired_vx = -WORLD.player_vx
        if (backward and (not forward)):
            self.desired_vx = WORLD.player_vx
        if jump:
            self.desired_vy = WORLD.player_vy
        if (right and (not left)):
            self.desired_vz = WORLD.player_vz
        if (left and (not right)):
            self.desired_vz = -WORLD.player_vz

    def move(self):
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP
        self.z += self.vz * TIMESTEP
        if WEBOTS_MODE:
            self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

    def step(self):
        self.x += self.vx * TIMESTEP
        self.y += self.vy * TIMESTEP
        self.z += self.vz * TIMESTEP
        if WEBOTS_MODE:
            self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

    def update(self):
        self.vy += GRAVITY * TIMESTEP
        if (self.y <= NUDGE * TIMESTEP):
            self.vy = self.desired_vy

        self.vx = self.desired_vx * self.dir
        self.vz = self.desired_vz
        self.move()

        if (self.y <= 0):
            self.y = 0;
            self.vy = 0;

        if (self.x * self.dir <= (WORLD.wall_width / 2 + self.r)):
            self.vx = 0;
            self.x = self.dir * (WORLD.wall_width / 2 + self.r)

        if (self.x * self.dir >= (WORLD.width / 2 - self.r)):
            self.vx = 0;
            self.x = self.dir * (WORLD.width / 2 - self.r)

        if WORLD.wall_depth >= self.r:
            if (self.z <= -(WORLD.wall_depth / 2 - self.r)):
                self.vz = 0;
                self.z = -(WORLD.wall_depth / 2 - self.r)

            if (self.z >= WORLD.wall_depth / 2 - self.r):
                self.vz = 0;
                self.z = WORLD.wall_depth / 2 - self.r
        else:
            if (self.z <= -(WORLD.wall_depth / 2)):
                self.vz = 0;
                self.z = -(WORLD.wall_depth / 2)

            if (self.z >= WORLD.wall_depth / 2):
                self.vz = 0;
                self.z = WORLD.wall_depth / 2
        if WEBOTS_MODE:
            self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

    def getObservation(self):
        return self.state.getObservation()

    def getMates(self, team):
        mates = team.copy()
        del mates[self.name]
        for i in mates:
            self.mates[i] = mates[i]

    def getObs(self, ball, opponents):
        obs = [self.x * self.dir, self.y, self.z, self.vx * self.dir, self.vy, self.vz]
        for i in self.mates:
            obs += [self.mates[i].x * self.dir, self.mates[i].y, self.mates[i].z,
                    self.mates[i].vx * self.dir, self.mates[i].vy, self.mates[i].vz]

        obs += [ball.x * self.dir, ball.y, ball.z, ball.vx * self.dir, ball.vy, ball.vz]

        op = opponents.getTeamState()
        for i in range(0, len(op), 3):
            op[i] *= -self.dir
        obs += op

        return np.array(obs)

    def getDist2(self, p):
        dz = p.z - self.z
        dy = p.y - self.y
        dx = p.x - self.x
        return (dx * dx + dy * dy + dz * dz)

    def isColliding(self, p):
        r = self.r + p.r
        if WORLD.depth != 0:
            # if distance is less than total radius and the depth, then colliding.
            return (r * r >= self.getDist2(p)) and (self.z * self.z <= WORLD.wall_depth * WORLD.wall_depth)
        else:
            return (r * r >= self.getDist2(p))

    def collision(self, p):
        abx = self.x - p.x
        aby = self.y - p.y
        abz = self.z - p.z
        abd = math.sqrt(abx * abx + aby * aby + abz * abz)
        if abd != 0:

            abx /= abd
            aby /= abd
            abz /= abd
            nx = abx
            ny = aby
            nz = abz
            abx *= NUDGE * 0.4
            aby *= NUDGE * 0.4
            abz *= NUDGE * 0.4

        else:
            abx = aby = abz = 1.7

        while (self.isColliding(p)):

            self.x += abx
            self.y += aby
            self.z += abz
            if WEBOTS_MODE:
                self.location.setSFVec3f([self.x * 0.1, self.y * 0.1, self.z * 0.1])

            p.x -= abx
            p.y -= aby
            p.z -= abz
            if WEBOTS_MODE:
                p.location.setSFVec3f([p.x * 0.1, p.y * 0.1, p.z * 0.1])

    def checkSide(self):
        return ((self.z + WORLD.max_depth / 2) // (WORLD.max_depth / 3)) == (
                self.side - 1)  # (self.z // (WORLD.max_depth/3)) + 1 == (self.side - 1) ?  # 3即n_agent，这儿也不调整了（


class BaselinePolicy:

    def __init__(self, agent=None, env=None):
        self.agent = agent
        self.env = env

    def predict(self, obs):
        if self.agent is None:
            actions = []
            for i in range(3):  # 3 try
                actions.append(self.env.action_space.sample())
        else:
            actions, _ = self.agent.predict(obs)
        return actions


class Game:

    def __init__(self, diff_identity=True, np_random=np.random, training=False, eval_mode=False):
        self.ball = None
        self.fenceStub = None
        self.team_left = None
        self.team_right = None
        self.delayScreen = None
        self.np_random = np_random
        self.training = training
        self.eval_mode = eval_mode
        self.reset()
        self.BallCollision = [0, 0, 0]
        self.NotSide = [0, 0, 0]
        self.match = 0
        self.n_mate = 3  # 改为3球员

    def reset(self):
        self.match = 0
        self.n_mate = 3
        self.fenceStub = Particle(0, WORLD.wall_height, 0, 0, 0, 0, WORLD.wall_width / 2, "FENCESTUB");
        if self.training:
            ball_vx = self.np_random.uniform(low=0, high=20)
        else:
            ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        ball_vz = self.np_random.uniform(low=0, high=10) * (WORLD.depth / WORLD.max_depth)
        # ball_vz = self.np_random.uniform(low=-10, high=10) * (WORLD.depth/WORLD.max_depth)
        self.ball = Particle(0, (WORLD.width / 4) - 1.5, 0, ball_vx, ball_vy, ball_vz, 0.5, "BALL");
        self.team_left = Team(-1, "BLUE", self.n_mate)
        self.team_right = Team(1, "YELLOW", self.n_mate)

        # self.team_left.updateState(self.ball, self.agent_right)
        # self.team_right.updateState(self.ball, self.agent_left)
        self.delayScreen = DelayScreen()

    def newMatch(self):
        self.match += 1

        if self.training:
            ball_vx = self.np_random.uniform(low=0, high=20)
        else:
            ball_vx = self.np_random.uniform(low=-20, high=20)
        ball_vy = self.np_random.uniform(low=10, high=25)
        if self.match % 2 == 0:
            ball_vz = self.np_random.uniform(low=0, high=10) * (WORLD.depth / WORLD.max_depth)
        else:
            ball_vz = self.np_random.uniform(low=-10, high=0) * (WORLD.depth / WORLD.max_depth)
        self.ball = Particle(0, (WORLD.width / 4) - 1.5, 0, ball_vx, ball_vy, ball_vz, 0.5, "BALL");
        self.delayScreen.reset()

    # 加了个新的惩罚，即（超出一定距离时）离球越远越大的惩罚
    def step(self, if_multi_role=IF_MULTI_ROLE):
        if if_multi_role:
            # 更新各队伍、球状态
            self.betweenGameControl()
            self.team_left.update()
            self.team_right.update()
            self.BallCollision = [0, 0, 0]
            self.NotSide = [0, 0, 0]

            if self.delayScreen.status():
                self.ball.applyAcceleration(0, WORLD.gravity, 0)
                self.ball.limitSpeed(0, WORLD.max_ball_v)
                self.ball.move()

            for agent in self.team_left.team:
                if (self.ball.isColliding(self.team_left.team[agent])):
                    self.ball.bounce(self.team_left.team[agent])
                mates = self.team_left.team[agent].mates
                for i in mates:
                    if (self.team_left.team[agent].isColliding(mates[i])):
                        self.team_left.team[agent].collision(mates[i])

            # 球是否与左边队伍的智能体相撞（对相撞的agent进行记录）
            for idx, agent in enumerate(self.team_right.team):

                mates = self.team_right.team[agent].mates
                if (self.ball.isColliding(self.team_right.team[agent])):
                    self.ball.bounce(self.team_right.team[agent])
                    self.BallCollision[int(agent[-1]) - 1] = 1
                    self.team_right.team[agent].ball_bonus = 0.1  # 接到球的计分

                dis_trd = 2 * self.ball.r
                abdistance = abs(self.ball.z - self.team_right.team[agent].z) - dis_trd
                if abdistance > 0:
                    self.team_right.team[agent].dis_malus = abdistance * (-0.00005)

                # print(self.team_right.team[agent].checkSide())
                # print(self.team_right.team[agent].name)

                # if (self.team_right.team[agent].checkSide()):
                #     self.team_right.team[agent].bonus = 0.001
                #     self.NotSide[int(agent[-1]) - 1] = 0
                # else:
                #     self.team_right.team[agent].malus = -0.001
                #     self.NotSide[int(agent[-1]) - 1] = 1

                for i in mates:
                    if (self.team_right.team[agent].isColliding(mates[i])):
                        self.team_right.team[agent].collision(mates[i])
                        self.team_right.team[agent].malus = -0.01
                        mates[i].malus = -0.01
                        # if (mates[i].checkSide()):
                        #     self.team_right.team[agent].malus = -0.01
                        # elif (self.team_right.team[agent].checkSide()):
                        #     mates[i].malus = -0.01

            # 球与栅栏
            self.fenceStub.z = self.ball.z
            if (self.ball.isColliding(self.fenceStub)):
                self.ball.bounce(self.fenceStub)

            result = -self.ball.checkEdges()  # 球在边界是反弹的，因此只有落地为判定标识

            if result == -1:
                agent_results = [-1 for i in range(3)]
            else:
                agent_results = [1 for i in range(3)]

            Bonus = []
            identity_list = []
            for agent in self.team_right.team:
                identity_tmp = self.team_right.team[agent].identity
                identity_list.append(identity_tmp)
                Bonus.append(
                    (self.team_right.team[agent].ball_bonus) * (IDENTITY_REWARD_WEIGHT[identity_tmp]["ball_bonus"])
                    + (self.team_right.team[agent].malus) * (IDENTITY_REWARD_WEIGHT[identity_tmp]["malus"])
                    + (self.team_right.team[agent].dis_malus) * (
                        IDENTITY_REWARD_WEIGHT[identity_tmp]["dis_malus"]))

            for agent in self.team_right.team:
                self.team_right.team[agent].malus = 0
                self.team_right.team[agent].bonus = 0
                self.team_right.team[agent].ball_bonus = 0
                self.team_right.team[agent].dis_malus = 0

            if (result != 0):
                self.newMatch()
                if result < 0:
                    self.team_left.emotion = "happy"
                    self.team_right.emotion = "sad"
                    self.team_right.life -= 1
                else:
                    self.team_left.emotion = "sad"
                    self.team_right.emotion = "happy"
                    self.team_left.life -= 1

                if self.eval_mode:
                    return [1 for i in range(3)]

                # print(agent_results)
                if result < 0:
                    return [agent_results[i] * IDENTITY_REWARD_WEIGHT[identity_list[i]]["result_fail"] + Bonus[i] for i
                            in range(len(Bonus))]
                else:
                    return [agent_results[i] * IDENTITY_REWARD_WEIGHT[identity_list[i]]["result_win"] + Bonus[i] for i
                            in range(len(Bonus))]

            if self.eval_mode:
                return [result for i in range(3)]

            return [result + i for i in Bonus]

        else:
            # 更新各队伍、球状态
            self.betweenGameControl()
            self.team_left.update()
            self.team_right.update()
            self.BallCollision = [0, 0, 0]
            self.NotSide = [0, 0, 0]

            if self.delayScreen.status():
                self.ball.applyAcceleration(0, WORLD.gravity, 0)
                self.ball.limitSpeed(0, WORLD.max_ball_v)
                self.ball.move()
            # 球是否与左边队伍的智能体相撞
            for agent in self.team_left.team:
                if (self.ball.isColliding(self.team_left.team[agent])):
                    self.ball.bounce(self.team_left.team[agent])
                mates = self.team_left.team[agent].mates
                for i in mates:  # 每个agent是否与同队伍的其他agent相撞
                    if (self.team_left.team[agent].isColliding(mates[i])):
                        self.team_left.team[agent].collision(mates[i])

            # 球是否与左边队伍的智能体相撞（对相撞的agent进行记录）
            for idx, agent in enumerate(self.team_right.team):

                mates = self.team_right.team[agent].mates
                if (self.ball.isColliding(self.team_right.team[agent])):
                    self.ball.bounce(self.team_right.team[agent])
                    self.BallCollision[int(agent[-1]) - 1] = 1

                if (self.team_right.team[
                    agent].checkSide()):
                    self.team_right.team[agent].bonus = 0.001
                    self.NotSide[int(agent[-1]) - 1] = 0
                else:
                    self.team_right.team[agent].malus = -0.001
                    self.NotSide[int(agent[-1]) - 1] = 1

                dis_trd = 2 * self.ball.r
                abdistance = abs(self.ball.z - self.team_right.team[agent].z) - dis_trd
                if abdistance > 0:
                    self.team_right.team[agent].dis_malus = abdistance * (-0.00005)

                for i in mates:
                    if (self.team_right.team[agent].isColliding(mates[i])):
                        self.team_right.team[agent].collision(mates[i])
                        if (mates[i].checkSide()):
                            self.team_right.team[agent].malus = -0.01
                        elif (self.team_right.team[agent].checkSide()):
                            mates[i].malus = -0.01

            # 球与栅栏
            self.fenceStub.z = self.ball.z
            if (self.ball.isColliding(self.fenceStub)):
                self.ball.bounce(self.fenceStub)

            result = -self.ball.checkEdges()  # 球在边界是反弹的，因此只有落地为判定标识

            if result == -1:
                if self.ball.z <= 0:
                    agent_results = [-2, -1, 0]
                else:
                    agent_results = [0, -1, -2]
            else:
                agent_results = [1 for i in range(3)]

            Bonus = []
            for agent in self.team_right.team:
                Bonus.append(
                    self.team_right.team[agent].bonus + self.team_right.team[agent].malus + self.team_right.team[
                        agent].dis_malus)

            for agent in self.team_right.team:
                self.team_right.team[agent].malus = 0
                self.team_right.team[agent].bonus = 0
                self.team_right.team[agent].dis_malus = 0

            if (result != 0):
                self.newMatch()
                if result < 0:
                    self.team_left.emotion = "happy"
                    self.team_right.emotion = "sad"
                    self.team_right.life -= 1
                else:
                    self.team_left.emotion = "sad"
                    self.team_right.emotion = "happy"
                    self.team_left.life -= 1

                if self.eval_mode:
                    return [result for i in range(3)]

                # print(agent_results)
                return [agent_results[i] + Bonus[i] for i in range(len(Bonus))]

            if self.eval_mode:
                return [result for i in range(3)]

            return [result + i for i in Bonus]

    def betweenGameControl(self):
        agent = [self.team_left, self.team_right]
        if (self.delayScreen.life > 0):
            pass
        else:
            agent[0].emotion = "happy"
            agent[1].emotion = "happy"


class TeamVolleyBot(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state', 'webots'],
        'video.frames_per_second': 50
    }

    action_table = [[0, 0, 0, 0, 0],  # NOOP
                    [1, 0, 0, 0, 0],  # FORWARD
                    [1, 0, 1, 0, 0],  # FORWARD JUMP
                    [1, 0, 0, 1, 0],  # FORWARD RIGHT
                    [1, 0, 0, 0, 1],  # FORWARD LEFT
                    [1, 0, 1, 1, 0],  # FORWARD JUMP RIGHT
                    [1, 0, 1, 0, 1],  # FORWARD JUMP LEFT
                    [0, 1, 0, 0, 0],  # BACKWARD
                    [0, 1, 1, 0, 0],  # BACKWARD JUMP
                    [0, 1, 0, 1, 0],  # BACKWARD RIGHT
                    [0, 1, 0, 0, 1],  # BACKWARD LEFT
                    [0, 1, 1, 1, 0],  # BACKWARD JUMP RIGHT
                    [0, 1, 1, 0, 1],  # BACKWARD JUMP LEFT
                    [0, 0, 1, 0, 0],  # JUMP
                    [0, 0, 1, 1, 0],  # JUMP RIGHT
                    [0, 0, 1, 0, 1],  # JUMP LEFT
                    [0, 0, 0, 1, 0],  # RIGHT
                    [0, 0, 0, 0, 1]]  # LEFT

    from_pixels = False
    survival_bonus = False  # Depreciated: augment reward, easier to train
    multiagent = True  # optional args anyways

    def __init__(self, training=False, update=False, n_agents=3, eval_mode=False):

        self.t = 0
        self.t_limit = 3000
        self.atari_mode = False
        self.num_envs = 1
        self.training = training
        self.update = update
        self.world = WORLD  # 初始或可设大一点，或者update快一点
        self.n_agents = n_agents
        self.ret = [0, 0, 0]
        self.BallCollision = [0, 0, 0]
        self.NotSide = [0, 0, 0]
        self.eval_mode = eval_mode

        if self.atari_mode:
            action_space = spaces.Discrete(3)
        else:
            action_space = spaces.MultiBinary(5)

        self.action_space = action_space

        if self.from_pixels:
            setPixelObsMode()
            observation_space = spaces.Box(low=0, high=255,
                                           shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3), dtype=np.uint8)
        else:
            high = np.array([np.finfo(np.float32).max] * (self.n_agents * 2 * 6 + 6))
            observation_space = spaces.Box(-high, high)

        self.observation_space = observation_space

        self.previous_rgbarray = None
        self.game = Game(training=self.training, diff_identity=True)
        self.ale = self.game.team_right

        self.policy = BaselinePolicy()
        self.viewer = None
        self.otherActions = None

        # # 创建按钮
        # self.buttons = []
        # self.left_buttons = []
        # self.right_buttons = []
        # self.create_controll_button()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game = Game(np_random=self.np_random, training=self.training, diff_identity=True)
        self.ale = self.game.team_right
        return [seed]

    def getObs(self):
        if self.from_pixels:
            obs = self.render(mode='state')
            self.canvas = obs
        else:
            obs = self.game.team_right.getObservations(self.game.ball, self.game.team_left)
        return obs

    def discreteToBox(self, n):
        actions = []
        for act in n:
            if isinstance(act, (list, tuple, np.ndarray)):
                if len(act) == 5:
                    actions.append(act)
        if len(actions) == len(n):
            return n

        for act in n:
            assert (int(act) == act) and (act >= 0) and (act < 18)

        return [self.action_table[act] for act in n]

    def step(self, actions, otherActions=None):
        if WEBOTS_MODE:
            self.draw_team_name()
            self.draw_lives(self.game.team_left.life, self.game.team_right.life)
            self.draw_time(int((self.t_limit - self.t) / 100))
            supervisor.step(TIME_STEP)
            global CUR_KEY
            CUR_KEY = game_keyboard.getKey()
            if ord('0') <= CUR_KEY <= ord(f'{2 * self.n_agents}'):
                num = CUR_KEY - ord('1')
                print(f"keyboard-num:{num}")
                for i in range(0, self.n_agents):
                    if i == num:
                        left_controlled_list[i] = True
                        print(f"{i} in left-team is switched on.")
                    else:
                        left_controlled_list[i] = False
                    if i + self.n_agents == num:
                        right_controlled_list[i] = True
                        print(f"{i} in right-team is switched on.")
                    else:
                        right_controlled_list[i] = False
                if num == -1:
                    pass

        done = False
        self.t += 1

        if self.otherActions is not None:
            otherActions = self.otherActions
        if otherActions is None:
            observations = self.game.team_left.getObservations(self.game.ball, self.game.team_right)

            for i in range(3):
                otherActions = self.policy.predict(observations)
        if self.atari_mode:
            actions = self.discreteToBox(actions)
            otherActions = self.discreteToBox(otherActions)

        self.game.team_left.setAction(otherActions)
        self.game.team_right.setAction(actions)

        for i in range(self.n_agents):
            self.game.team_left.team[self.game.team_left.name + str(i + 1)].controlled_by_board = left_controlled_list[
                i]
        for i in range(self.n_agents):
            self.game.team_right.team[self.game.team_right.name + str(i + 1)].controlled_by_board = \
                right_controlled_list[i]

        reward = self.game.step()
        self.ret = [reward[i] + self.ret[i] for i in range(len(reward))]
        self.BallCollision = [self.game.BallCollision[i] + self.BallCollision[i] for i in range(3)]
        self.NotSide = [self.game.NotSide[i] + self.NotSide[i] for i in range(3)]

        obs = self.getObs()

        if self.t >= self.t_limit:
            done = True
        if self.game.team_left.life <= 0 or self.game.team_right.life <= 0:
            done = True

        otherObs = None
        if self.multiagent:
            if self.from_pixels:
                otherObs = cv2.flip(obs, 1)
            else:
                otherObs = self.game.team_left.getObservations(self.game.ball, self.game.team_right)

        info = {
            'ale.lives': self.game.team_right.lives(),
            'ale.otherLives': self.game.team_left.lives(),
            'otherObs': otherObs,
            'state': self.game.team_right.getObservations(self.game.ball, self.game.team_left),
            'otherState': self.game.team_left.getObservations(self.game.ball, self.game.team_right),
            'otherAction': otherActions,
            'EnvDepth': self.world.depth
        }
        return obs, reward, done, [info, copy.copy(info)]

    def init_game_state(self):
        self.t = 0
        self.ret = [0, 0, 0]
        self.BallCollision = [0, 0, 0]
        self.NotSide = [0, 0, 0]

        self.game.reset()
        if self.eval_mode:
            self.game.eval_mode = True
        else:
            self.game.eval_mode = False
        if self.training:
            self.game.training = True
        else:
            self.game.training = False

        if not self.update:
            self.world.update = False
            self.world.setup()

    def reset(self):
        if WEBOTS_MODE:
            self.draw_time(int(self.t_limit / 100))
        self.init_game_state()
        return self.getObs()

    def get_action_meanings(self):
        return [self.atari_action_meaning[i] for i in self.atari_action_set]

    def update_world(self):
        if self.update:
            self.world.update_world()
        if not self.world.update:
            self.update = False

    def draw_team_name(self):
        supervisor.setLabel(
            0,
            "Yellow",
            0.76 - (len("Yellow") * 0.01),
            0.01,
            0.1,
            0xFFFF00,
            0.0,
            "Tahoma",
        )
        supervisor.setLabel(
            1,
            "Blue",
            0.05,
            0.01,
            0.1,
            0x0000FF,
            0.0,
            "Tahoma",  # Fon
        )

    def draw_lives(self, left_life: int, right_life: int):
        supervisor.setLabel(
            2,
            "remaining lives: " + str(right_life - 1),
            0.7,
            0.05,
            0.1,
            0xFFFF00,
            0.0,
            "Tahoma",
        )
        supervisor.setLabel(
            3,  # LabelId
            "remaining lives: " + str(left_life - 1),
            0.05,
            0.05,
            0.1,
            0x0000FF,
            0.0,
            "Tahoma",
        )

    def draw_time(self, time: int):
        supervisor.setLabel(
            4,
            "00:" + str(time),
            0.45,
            0.01,
            0.1,
            0x000000,
            0.0,
            "Arial",
        )

    def draw_event_messages(self, messages):
        if messages:
            supervisor.setLabel(
                5,
                "New match!",
                0.01,
                0.95 - ((len(messages) - 1) * 0.025),
                0.05,
                0xFFFFFF,
                0.0,
                "Tahoma",
            )

    # ......
    # def create_controll_button(self):
    #     # 创建两个窗口
    #     window1 = supervisor.openWindow("Window 1", 100, 100, 200, 200)
    #     window2 = supervisor.openWindow("Window 2", 400, 100, 200, 200)
    #
    #     for i in range(self.n_agents):
    #         # 在窗口1中创建蓝色按钮
    #         button1 = supervisor.buttonCreate(window1, "Button " + str(i + 1))
    #         button1.setColor(0x0000FF)  # 设置按钮颜色为蓝色
    #         button1.setClickedCallback(self.button_callback)  # 设置按钮回调函数
    #         self.left_buttons.append(button1)
    #         self.buttons.append(button1)
    #
    #         # 在窗口2中创建黄色按钮
    #         button2 = supervisor.buttonCreate(window2, "Button " + str(i + 1))
    #         button2.setColor(0xFFFF00)  # 设置按钮颜色为黄色
    #         button2.setClickedCallback(self.button_callback)  # 设置按钮回调函数
    #         self.left_buttons.append(button2)
    #         self.buttons.append(button2)
    #
    # def button_callback(self,button):
    #     # 读取按钮的值
    #     value = button.getValue()
    #
    #     if value == 1:
    #         # 按下了一个按钮，弹起其他按钮
    #         for other_button in self.buttons:
    #             if other_button != button:
    #                 other_button.setValue(0)
    #     return value
