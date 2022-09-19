import os
import socket
import math
import random
import numpy as np
import win32api
try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')
import time

UR5name = 'CR5_joint'
rad = math.pi / 180
force_con = 1
f_aim = [0, 0, 60, 0, 0, 0]

class peg_in_hole_env(object):
    def __init__(self, host):
        # self.conn = None
        # self.addr = None
        # self.clientID = -1
        # self.joint_handle = []
        # self.dummy_handle = 0
        # self.force_handle = 0

        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', host))
        self.server.listen()
        self.conn, self.addr = self.server.accept()
        print(self.conn)
        # while 1:
        sim.simxFinish(-1) # just in case, close all opened connections
        self.clientID = sim.simxStart('127.0.0.1',19989,True,True,5000,5) # Connect to CoppeliaSim
        print(self.clientID)
            # if self.clientID != -1:
            #     break
        if self.clientID != -1:
            print ('Connected to remote API server')
            # Now try to retrieve data in a blocking fashion (i.e. a service call):
            # 获取各关节句柄
            self.joint_handle = []
            for i in range(6):
                _, self.handle = sim.simxGetObjectHandle(self.clientID, UR5name + str(i+1), sim.simx_opmode_blocking)
                self.joint_handle.append(self.handle)
            _, self.dummy_handle = sim.simxGetObjectHandle(self.clientID, 'CR5_target', sim.simx_opmode_blocking)
            _, self.force_handle = sim.simxGetObjectHandle(self.clientID, 'CR5_connection', sim.simx_opmode_blocking)

    def Stop(self):
        self.conn.close()
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)

    def Closs(self, path):
        sim.simxCloseScene(self.clientID, sim.simx_opmode_blocking)
        sim.simxLoadScene(self.clientID, path, -1, sim.simx_opmode_blocking)

    def ArrtoStr(self, arr):
        str1 = ','.join(str(i) for i in arr)
        return str1

    def GetTCPpos(self):
        _, state_postation = sim.simxGetObjectPosition(self.clientID, self.dummy_handle, -1, sim.simx_opmode_blocking)
        _, state_orientation = sim.simxGetObjectOrientation(self.clientID, self.dummy_handle, self.dummy_handle, sim.simx_opmode_blocking)
        data = np.append(state_postation, [i / rad for i in state_orientation])
        return data

    def Getdate(self):
        _, state_postation = sim.simxGetObjectPosition(self.clientID, self.dummy_handle, -1, sim.simx_opmode_blocking)
        _, state_orientation = sim.simxGetObjectOrientation(self.clientID, self.dummy_handle, -1, sim.simx_opmode_blocking)
        _, _1, force, torque = sim.simxReadForceSensor(self.clientID, self.force_handle, sim.simx_opmode_oneshot)
        data = np.append(np.append(np.append(state_postation, [i / rad for i in state_orientation]), force), torque)
        return data

    def Getdatej(self, f):
        pos = [0,0,0,0,0,0]
        for i in range(6):
            _, pos[i] = sim.simxGetJointPosition(self.clientID, self.joint_handle[i],sim.simx_opmode_blocking)
        _, _1, force, torque = sim.simxReadForceSensor(self.clientID, self.force_handle, sim.simx_opmode_oneshot)
        # data = np.append(np.append(pos, force), torque)
        data = np.append(pos, f)
        return data

    def Getdatej_f(self):
        pos = [0,0,0,0,0,0]
        for i in range(6):
            _, pos[i] = sim.simxGetJointPosition(self.clientID, self.joint_handle[i],sim.simx_opmode_blocking)
        _, _1, force, torque = sim.simxReadForceSensor(self.clientID, self.force_handle, sim.simx_opmode_oneshot)
        data = np.append(np.append(pos, force), torque)
        return data

    def TCP_Move(self, pos):
        sim.simxSetObjectPosition(self.clientID, self.dummy_handle, -1, pos[:3], sim.simx_opmode_oneshot)
        sim.simxSetObjectOrientation(self.clientID, self.dummy_handle, self.dummy_handle, [i * math.pi / 180 for i in pos[3:]],sim.simx_opmode_oneshot)
    
    def Move(self, pos):
        sim.simxSetObjectPosition(self.clientID, self.dummy_handle, -1, pos[:3], sim.simx_opmode_oneshot)
        sim.simxSetObjectOrientation(self.clientID, self.dummy_handle, -1, [i * math.pi / 180 for i in pos[3:]],sim.simx_opmode_oneshot)

    def reset(self):
        start_pos = [0.3708456543505616,0.10773659350276255, 0.6231, 0, 180, 0]
        self.Move(start_pos)
        time.sleep(0.5)
        # np.random.seed(self.seed)
        random_x = random.uniform(0, 4)
        # 生成-10至10之间的随机数
        random_x = (random_x - 2) / 2000
        # np.random.seed(self.seed)
        random_y = random.uniform(0, 4)
        # 生成-10至10之间的随机数
        random_y = (random_y - 2) / 2000
        # 生成-0.05至0.05之间的随机数
        # np.random.seed(self.seed)
        random_alaph = random.uniform(0, 6)
        random_alaph = (random_alaph - 3) / 1.5
        # 生成-0.05至0.05之间的随机数
        # np.random.seed(self.seed)
        random_beta = random.uniform(0, 6)
        random_beta = (random_beta - 3) / 1.5
        start_pos = self.GetTCPpos()
        # print(start_pos)
        random_pos = [start_pos[0] + random_x, start_pos[1] + random_y, start_pos[2], start_pos[3] + random_alaph,
                      start_pos[4] + random_beta, start_pos[5]]
        # print(random_pos)
        self.TCP_Move(random_pos)
        # time.sleep(0.5)
        random_pos1 = self.Getdate()
        self.move_z(random_pos1[:6])
        # random_state = self.Getdatej_f()
        random_state = self.Getdate()
        # print(random_state)
        return random_state

    def move_z(self, pos):
        while True:
            pos[2] = pos[2] - 0.0001
            self.Move(pos)
            time.sleep(0.0001)
            if pos[2] < 0.624:
                break

    def move_zp(self, pos):
        print("回到初始位置")
        self.Move(pos)
        time.sleep(0.1)
        while True:
            self.Move(pos)
            pos[2] = pos[2] + 0.002
            time.sleep(0.0001)
            if pos[2] > 0.624:
                break

    def next_step(self, action, step, step_max, interval_):
        action[6] = np.clip(action[6]/4 + 0.5, 0, 1)
        interval = action[6] * 0.001
        divisor1 = 0.0003 + interval/2
        divisor2 = 0.0003 + interval/2
        # divisor1 = 0.0002 + interval * 5 /(step + 5)
        # divisor2 = 0.0002 + interval * 5 /(step + 5)
        divisor3 = 0.00025
        # divisor3 = 0
        divisor = 0.1 + 100 * interval
        # divisor = 0
        limit = 2
        # state = fc.force_control(force_con, f_aim, state1[6:], state1[:6], k)
        # print("力", state1[6:], "位置", state1[:6])
        action[0] = (np.clip(action[0], -limit, limit) * divisor1)
        action[1] = (np.clip(action[1], -limit, limit) * divisor2)
        action[2] = (np.clip(action[2]+2, 0, 4) * divisor3)
        action[3] = (np.clip(action[3], -limit, limit) * divisor)
        action[4] = (np.clip(action[4], -limit, limit) * divisor)
        action[5] = (np.clip(action[5], -limit, limit) * divisor)

        a=['{:.10f}'.format(action[0]),'{:.10f}'.format(action[1]),'{:.10f}'.format(action[2]),'{:.10f}'.format(action[3]),
        '{:.10f}'.format(action[4]),'{:.10f}'.format(action[5])]

        # print(a)

        self.conn.send(self.ArrtoStr(a).encode('utf-8'))
        # print(a)
        f = self.conn.recv(1024).decode('utf-8')
        f = list(f.split(","))
        f = [float(x) for x in f]
        time.sleep(1/8)

        state = self.Getdate()
        done, reward = self.reward_fun(action[2], state, f, step, step_max, interval, interval_)
        state3 = self.Getdatej(f)

        if (f[0] > 150) or (f[1] > 150) or (f[2] > 130):
            done = 2
            print("force chao xian****************************")
            # reward = reward - 2
            print('done=' + str(done) + '**************************')
            return state, state3, reward, done, interval

        print('done=' + str(done) + '**************************')
        return state, state3, reward, done, interval

    def reward_fun(self, action, state, f, step, step_max, interval, interval_):
        # r1 = 0
        # if step > 5:
        r1 = - step / step_max
            # print("step：", r1)
        r2 = abs(0.572 - state[2]) / 0.05
        # if abs(state[6])>50 or abs(state[7])>50 or abs(state[8]-20)>50:
        #     r3 = 0
        # else:
        # r3 = min((1-abs(state[6]) / 50), (1-abs(state[7]) / 50), (1-abs(state[8]-20) / 50))
        rf = 0.3 * (1-abs(f[0]) / 100) + 0.3 * (1-abs(f[1]) / 100) + 0.4 * (1-abs(f[2]) / 100)
        rt = 0.5 * (1-abs(f[3]) / 10) + 0.5 * (1-abs(f[4]) / 10)
        r3 = 0.3 * rt + 0.7 * rf
        # print("力：", r3)
        # if abs(interval_ - interval) * 1000 > 0.2:
        #     r4 = - 0.2
        # else:
        r4 = - abs(interval_ - interval) * 1000
        # print("间隙：", r4)
        r5 = abs(action)/0.001
        # print("z:", action, "reward:", r5)
        # print("z：", r5)
        # r = 5 * r1 + 7 * r3 + 5 * r4 + 3 * r5
        r = 5 * r1 + 1.5 * r2 + 7 * r3 + 5 * r4 + 1.5 * r5
        # r = 5 * r1 + 10 * r3 + 5 * r4
        # r = 0.5 * r1 + 0.2 * r2 + 0.4 * r3 + 0.5 * r4 + 0.4 * r5
        sw = self.Getdate()
        if sw[2] < 0.572:
            done = 1
            print("装配成功！！！！！！！！！！！！！！！！！！！！！！！！！！")
        else:
            done = 0
        # print("奖励值" + str(r))
        return done, r