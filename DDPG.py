"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_DDPG.py --train/test

"""

import argparse
import os
import time
from copy import deepcopy

import pandas as pd

import matplotlib.pyplot as plt

import open_close
import simulation_env_ddpg as env
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
import pyautogui
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'Pendulum-v0'    # environment name
RANDOMSEED = 1              # random seed

LR_A = 0.0001                # learning rate for actor
LR_C = 0.0002                # learning rate for critic
GAMMA = 0.9                 # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 1500      # size of replay buffer
BATCH_SIZE = 128             # update batchsize

MAX_EPISODES = 1000          # total number of episodes for training
MAX_EP_STEPS = 30           # total number of steps for each episode
TEST_PER_EPISODES = 100      # test the model per episodes
VAR = 1                     # control exploration

history = {'episode': [], 'Episode_reward': [], 'step': [], 'time': []}
force_torque = {'episode': [], 'x_force': [], 'y_force': [], 'z_force': [], 'x_torque': [], 'y_torque': [],
                'z_torque': [], 'reward': [], 'interval': []}
tra = {'episode': [], 'step': [], 'x': [], 'y': [], 'z': [], 'alph': [], 'beta': [], 'gama': []}

###############################  DDPG  ####################################

class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, a_dim, s_dim):
        # memory????????????????????????????????????
        # ????????????MEMORY_CAPACITY???s_dim * 2 + a_dim + 1??????????????????state?????????action????????????reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim = a_dim, s_dim

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # ??????actor???????????????s?????????a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=100, act=tf.nn.sigmoid, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            # a = tl.layers.Lambda(lambda x: 2 * x, name='Actor' + name)(x)            #?????????????????????tanh??????????????????[-1,1]????????????????????????
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)
            # return a

        #??????Critic???????????????s???a?????????Q???
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        #?????????????????????????????????????????????????????????
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        #??????actor_target???????????????actor???????????????????????????
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        #??????critic_target???????????????actor???????????????????????????
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        #??????ema??????????????????
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)


    def ema_update(self):
        """
        ??????????????????
        """
        # ???????????????????????????????????????????????????????????????????????????ema.average???
        paras = self.actor.trainable_weights + self.critic.trainable_weights    #??????????????????????????????actor???critic???
        self.ema.apply(paras)                                                   #???????????????????????????
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       # ?????????????????????

    # ??????????????????s??????????????????a
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        return self.actor(np.array([s], dtype=np.float32))[0]

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #??????BATCH_SIZE????????????
        bt = self.memory[indices, :]                    #??????indices???????????????bt??????????????????
        bs = bt[:, :self.s_dim]                         #???bt????????????s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #???bt????????????a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #???bt????????????r
        bs_ = bt[:, -self.s_dim:]                       #???bt????????????s'

        # Critic???
        # Critic?????????DQN???????????????target??????argmax????????????critic_target??????????????????
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor???
        # Actor???????????????????????????Q?????????
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()


    # ??????s???a???r???s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # ??????s???s_,??????????????????????????????
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        #???s, a, [r], s_????????????
        transition = np.hstack((s, a, [r], s_))

        #pointer??????????????????????????????????????????
        #index?????????????????????????????????????????????
        #???????????????????????????MEMORY_CAPACITY???????????????index???????????????????????????
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #???transition????????????s, a, [r], s_????????????
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self, path):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(path + 'ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5(path + 'ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5(path + 'ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5(path + 'ddpg_critic_target.hdf5', self.critic_target)

    def load_ckpt(self, path):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order(path + 'ddpg_actor.hdf5'.encode('utf-8').decode('utf-8'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(path + 'ddpg_actor_target.hdf5'.encode('utf-8').decode('utf-8'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(path + 'ddpg_critic.hdf5'.encode('utf-8').decode('utf-8'),
                                               self.critic)
        tl.files.load_hdf5_to_weights_in_order(path + 'ddpg_critic_target.hdf5'.encode('utf-8').decode('utf-8'), self.critic_target)

    def save_history(self, history, name, path):
        if not os.path.exists(path):
            os.makedirs(path)
        name = os.path.join(path, name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')


if __name__ == '__main__':
    print("_____________________440_________________________")
    interval_ = 0.00044
    minloss = -10000
    writer = SummaryWriter("runs/" + "/DDPG/" + "440/")
    model_path = './model/' + "/DDPG/" + '440/'
    csv_path = './history/DDPG/' + '440/'
    open_close.Open('\losse_match_ppo.ttt')
    time.sleep(7)
    # print(pyautogui.position())
    pyautogui.moveTo(1159, 574, duration=0.25)  # ??????????????????????????????duration??????????????????
    pyautogui.click()
    time.sleep(2)
    pyautogui.moveTo(880, 66, duration=0.25)  # ??????????????????????????????duration??????????????????
    pyautogui.click()
    env = env.peg_in_hole_env(5006)

    # #???????????????
    # # writer = SummaryWriter("runs/" + "DDPG_test_128_relu_Multinorm")
    # env = env.peg_in_hole_env()

    #??????????????????????????????????????????????????????
    s_dim = 12
    a_dim = 7
    # a_bound = env.action_space.high

    #???DDPG??????
    ddpg = DDPG(a_dim, s_dim)
    #
    # #???????????????
    # if args.train:  # train
    #     i = 0
    #     b = 1.1
    #     updata, up_time, keep, keep_r, l, ep = 0, 0, 0, 0, 0, 0
    #     reward_buffer = []      #??????????????????EP???reward???????????????
    #     t0 = time.time()        #????????????
    #     while i < MAX_EPISODES:
    #     # for i in range(MAX_EPISODES):
    #         t1 = time.time()
    #         i += 1
    #         keep += 1
    #         s = env.reset()
    #         ep_reward = 0       #????????????EP???reward
    #         # VAR = b - i * 2/MAX_EPISODES
    #         # if VAR < 0.3:
    #         #     VAR = 0.3
    #         for j in range(MAX_EP_STEPS):
    #             # Add exploration noise
    #             a = ddpg.choose_action(s)       #???????????????????????????actor?????????a??????
    #             # ?????????????????????????????????????????????????????????????????????
    #             # ?????????????????????a????????????VAR???????????????????????????????????????????????????????????????a
    #             # ??????a??????????????????a???????????????????????????a????????????????????????????????????VAR?????????????????????????????????????????????VAR???????????????a????????????
    #             # ??????????????????
    #             a = np.clip(np.random.normal(a, VAR), -1, 1)
    #             # print("???????????????",a)
    #             # ?????????????????????
    #             # s_, r, done = env.next_step(a, j, MAX_EP_STEPS)
    #             state, s_, r, done, interval = env.next_step(deepcopy(a) * 2, j, MAX_EP_STEPS, interval_)
    #             # print("???????????????",a)
    #
    #             if (done == 1 and j <= 2) or (j == 0 and state[2] < 0.6):
    #                 print("????????????##############################################")
    #                 keep = keep - 1
    #                 i = i - 1
    #                 s2 = env.Getdate()
    #                 pos = [0.3708456543505616, 0.10773659350276255, s2[2], 0, 180, 0]
    #                 env.move_zp(pos)
    #                 break
    #             updata = updata + 1
    #
    #             # ??????s???a???r???s_
    #             ddpg.store_transition(s, a, r / 10, s_)
    #
    #             # ?????????????????????????????????????????????
    #             if updata >= MEMORY_CAPACITY:
    #                 print("???????????????????????????????????????????????????????????????????????????")
    #                 ddpg.learn()
    #
    #             #??????????????????
    #             s = s_
    #             ep_reward = 0.9 * ep_reward + r  #????????????EP??????reward
    #
    #             force_torque['episode'].append(ep)
    #             force_torque['x_force'].append(s[6])
    #             force_torque['y_force'].append(s[7])
    #             force_torque['z_force'].append(s[8])
    #             force_torque['x_torque'].append(s[9])
    #             force_torque['y_torque'].append(s[10])
    #             force_torque['z_torque'].append(s[11])
    #             force_torque['reward'].append(r)
    #             force_torque['interval'].append(interval)
    #
    #             tra['episode'].append(ep)
    #             tra['step'].append(j)
    #             tra['x'].append(state[0])
    #             tra['y'].append(state[1])
    #             tra['z'].append(state[2])
    #             tra['alph'].append(state[3])
    #             tra['beta'].append(state[4])
    #             tra['gama'].append(state[5])
    #
    #             writer.add_scalar("interval", interval, updata)
    #             writer.add_scalar("r", r, updata)
    #             writer.add_scalar("x", a[0], updata)
    #             writer.add_scalar("y", a[1], updata)
    #             writer.add_scalar("z", a[2], updata)
    #             writer.add_scalar("alph", a[3], updata)
    #             writer.add_scalar("beta", a[4], updata)
    #             writer.add_scalar("gama", a[5], updata)
    #             writer.add_scalar("inter", a[6], updata)
    #
    #             if j == MAX_EP_STEPS - 1:
    #                 done = True
    #                 print("??????????????????????????????????????????????????????????????????????????????????????????")
    #
    #             if done:
    #                 t1 = time.time() - t0
    #                 # data = json.dumps(str(1))
    #                 # data = data.encode('UTF-8')
    #                 # soc.sendall(data)
    #                 s2 = env.Getdate()
    #                 pos = [0.3708456543505616, 0.10773659350276255, s2[2], 0, 180, 0]
    #                 env.move_zp(pos)
    #                 break
    #
    #         if (done == 1 and j <= 2) or (j == 0 and state[2] < 0.6):
    #             continue
    #
    #
    #         # if i == 0:
    #         #     reward_buffer.append(ep_reward)
    #         # else:
    #         #     reward_buffer.append(reward_buffer[-1] * 0.9 + ep_reward * 0.1)
    #         print(
    #             'Episode: {}/{}  | Episode Reward: {:.4f} | Step: {} | Running Time: {:.4f}'.format(
    #             i, MAX_EPISODES, ep_reward, j,
    #             t1
    #             ), end='\n'
    #             )
    #         history['episode'].append(i)
    #         history['Episode_reward'].append(ep_reward)
    #         history['step'].append(j)
    #         history['time'].append(time.time() - t0)
    #         writer.add_scalar("Reward", ep_reward, i)
    #         writer.add_scalar("Step", j, i)
    #
    #         keep_r += ep_reward
    #
    #         if keep == 19:
    #             if keep_r > minloss:
    #                 l += 1
    #                 minloss = keep_r
    #                 ddpg.save_ckpt(model_path)
    #                 print("keep", l, "times")
    #             keep = 0
    #             keep_r = 0
    #
    #     ddpg.save_history(history, 'history.csv', csv_path)
    #     ddpg.save_history(force_torque, 'force_torque.csv', csv_path)
    #     ddpg.save_history(tra, 'tra.csv', csv_path)
        
    ###############################TEST#########################################
    history = {'episode': [], 'Episode_reward': [], 'step': [], 'time': []}
    test = {'cishu': [], 'episode': [], 'Episode_reward': [], 'step': [], 'time': [], 'success': []}
    force_torque = {'episode': [], 'x_force': [], 'y_force': [], 'z_force': [], 'x_torque': [], 'y_torque': [],
                    'z_torque': [], 'reward': [], 'interval': []}
    tra = {'episode': [], 'step': [], 'x': [], 'y': [], 'z': [], 'alph': [], 'beta': [], 'gama': []}
    tra_test = {'cishu': [], 'episode': [], 'step': [], 'x': [], 'y': [], 'z': [], 'alph': [], 'beta': [],
                'gama': []}
    l = 0
    writer_test = SummaryWriter("runs/" + "/DDPG/" + "test/")

    for m in range(5):
        success = 0
        ddpg.load_ckpt(model_path)
        t0 = 0
        # for t0 in range(TEST_LEN):
        while t0 <= TEST_PER_EPISODES:
            t0 += 1
            s = env.reset()
            t1 = time.time()
            ep_r = 0
            for i in range(MAX_EP_STEPS):
                # env.reset()
                a = ddpg.choose_action(s)
                a = np.clip(np.random.normal(a, VAR), -1, 1)
                state, s, r, done, interval = env.next_step(deepcopy(a), i, MAX_EP_STEPS, interval_)

                if (done == 1 and i <= 2) or (i == 0 and state[2] < 0.6):
                    print("????????????##############################################")
                    t0 = t0 - 1
                    s2 = env.Getdate()
                    pos = [0.3708456543505616, 0.10773659350276255, s2[2], 0, 180, 0]
                    env.move_zp(pos)
                    break
                l = l + 1

                force_torque['episode'].append(t0)
                force_torque['x_force'].append(s[6])
                force_torque['y_force'].append(s[7])
                force_torque['z_force'].append(s[8])
                force_torque['x_torque'].append(s[9])
                force_torque['y_torque'].append(s[10])
                force_torque['z_torque'].append(s[11])
                force_torque['reward'].append(r)
                force_torque['interval'].append(interval)

                writer_test.add_scalar("interval", interval, l)
                writer_test.add_scalar("r", r, l)
                writer_test.add_scalar("x", a[0], l)
                writer_test.add_scalar("y", a[1], l)
                writer_test.add_scalar("z", a[2], l)
                writer_test.add_scalar("alph", a[3], l)
                writer_test.add_scalar("beta", a[4], l)
                writer_test.add_scalar("gama", a[5], l)
                writer_test.add_scalar("inter", a[6], l)

                if i == MAX_EP_STEPS - 1:
                    # success = success - 1
                    r = r - 1
                    done = 2
                    print("??????????????????????????????????????????????????????????????????????????????????????????")

                if done == 1:
                    success += 1
                    s2 = env.Getdate()
                    pos = [0.3708456543505616, 0.10773659350276255, s2[2], 0, 180, 0]
                    env.move_zp(pos)
                    break

                if done == 2:
                    s2 = env.Getdate()
                    pos = [0.3708456543505616, 0.10773659350276255, s2[2], 0, 180, 0]
                    env.move_zp(pos)
                    break

                ep_r = 0.9 * ep_r + r

                tra_test['cishu'].append(m)
                tra_test['episode'].append(t0)
                tra_test['step'].append(i)
                tra_test['x'].append(state[0])
                tra_test['y'].append(state[1])
                tra_test['z'].append(state[2])
                tra_test['alph'].append(state[3])
                tra_test['beta'].append(state[4])
                tra_test['gama'].append(state[5])

            if (done == 1 and i <= 2) or (i == 0 and state[2] < 0.6):
                continue

            print(
                'cishu: {}/{} | Episode: {}/{}  | Episode Reward: {:.4f}  | Running step: {}  | Running Time: {:.4f} | success: {}'
                    .format(m, 5, t0, TEST_PER_EPISODES, ep_r, i, time.time() - t1, success))
            test['cishu'].append(m)
            test['episode'].append(t0)
            test['Episode_reward'].append(ep_r)
            test['step'].append(i)
            test['time'].append(time.time() - t1)
            test['success'].append(success)

            writer_test.add_scalar("Reward", ep_r, m * 30 + t0)
            writer_test.add_scalar("Step", i, m * 30 + t0)

        # ddpg.save_history(test, 'test' + str(m) + '.csv')
        ddpg.save_history(tra_test, 'tra_test.csv', csv_path)
    ddpg.save_history(test, 'test.csv', csv_path)
    ddpg.save_history(force_torque, 'force_torque.csv', csv_path)