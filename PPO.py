"""
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
--------------
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_PPO.py --train/test

"""

import argparse
import os
import random
import time
import pandas as pd
import simulation_env as env
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorboardX import SummaryWriter
import tensorlayer as tl
from copy import deepcopy

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_NAME = 'Pendulum-v0'        # environment name
RANDOMSEED = 1                  # random seed
TEST_LEN = 30
EP_MAX = 1000                   # total number of episodes for training
EP_LEN = 30                    # total number of steps for each episode
GAMMA = 0.9                     # reward discount
A_LR = 0.00001                   # learning rate for actor
C_LR = 0.0002                   # learning rate for critic
BATCH = 16                      # update batchsize
A_UPDATE_STEPS = 10     # actor update steps
C_UPDATE_STEPS = 10     # critic update steps
S_DIM, A_DIM = 12, 7     # state dimension, action dimension
EPS = 1e-10              # epsilon
history = {'episode': [], 'Episode_reward': [], 'step': [], 'time': []}
test = {'cishu': [], 'episode': [], 'Episode_reward': [], 'step': [], 'time': [], 'success': []}
force_torque = {'episode': [], 'x_force': [], 'y_force': [], 'z_force': [], 'x_torque': [], 'y_torque': [], 'z_torque': [], 'reward': [], 'interval': []}
tra = {'episode': [],'step': [], 'x': [], 'y': [], 'z': [], 'alph': [], 'beta': [],'gama': []}
tra_test = {'cishu': [], 'episode': [],'step': [], 'x': [], 'y': [], 'z': [], 'alph': [], 'beta': [],'gama': []}

# ??????????????????PPO1???PPO2?????????????????????
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty  PPO1
    dict(name='clip', epsilon=0.1),                 # Clipped surrogate objective, find this is better  PPO2
            ][1]                                                # choose the method for optimization

###############################  PPO  ####################################

class PPO(object):
    '''
    PPO ???
    '''

    def __init__(self):

        # ??????critic?????????
        # ??????state?????????V???
        tfs = tl.layers.Input([None, S_DIM], tf.float32, 'state')
        l1 = tl.layers.Dense(100, tf.nn.sigmoid)(tfs)
        v = tl.layers.Dense(1)(l1)
        self.critic = tl.models.Model(tfs, v)
        self.critic.train()

        # ??????actor?????????
        # actor?????????actor ??? actor_old??? actor_old????????????????????????????????????????????????
        # ?????????state?????????????????????????????????mu???sigma
        self.actor = self._build_anet('pi', trainable=True)
        self.actor_old = self._build_anet('oldpi', trainable=False)
        self.actor_opt = tf.optimizers.Adam(A_LR)
        self.critic_opt = tf.optimizers.Adam(C_LR)

    def a_train(self, tfs, tfa, tfadv):
        '''
        ??????????????????(policy network)
        '''
        # ?????????s???a???td-error????????????AC???????????????
        tfs = np.array(tfs, np.float32)         #state
        tfa = np.array(tfa, np.float32)         #action
        tfadv = np.array(tfadv, np.float32)     #td-error

        with tf.GradientTape() as tape:

            # ??????????????????????????????????????????
            # ????????????????????????????????????????????????????????????pi???oldpi???
            mu, sigma = self.actor(tfs)
            pi = tfp.distributions.Normal(mu, sigma)

            mu_old, sigma_old = self.actor_old(tfs)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            # ???????????????????????????????????????a??????????????????
            # ??????(oldpi.prob(tfa) + EPS)?????????????????????import-sampling???????????????????????????
            # ??????????????????????????????pi.prob(tfa)???????????????????????????????????????????????????????????????(oldpi.prob(tfa) + EPS)???
            # ???AC??????PG???????????????1,0??????????????????????????????????????????1or0?????????
            # ???PPO?????????????????????oldpi.prob(tfa)??????????????????????????????or?????????????????????
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + EPS)
            # ratio = tf.exp(pi.log_prob(tfa) - oldpi.log_prob(tfa))
            # ????????????????????????????????????????????????
            surr = ratio * tfadv

            # ?????????????????????????????????????????????
            # PPO1
            if METHOD['name'] == 'kl_pen':
                tflam = METHOD['lam']
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                kl_mean = tf.reduce_mean(kl)
                aloss = -(tf.reduce_mean(surr - tflam * kl))
            # PPO2???
            # ???????????????????????????????????????
            else:  # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(ratio * tfadv,  #surr
                               tf.clip_by_value(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * tfadv)
                )
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        # writer.add_scalar("Loss", aloss, ep)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if METHOD['name'] == 'kl_pen':
            return kl_mean
        return aloss

    def update_old_pi(self):
        '''
        ??????actor_old?????????
        '''
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)

    def c_train(self, tfdc_r, s):
        '''
        ??????Critic??????
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32) #tfdc_r???????????????PG?????????G????????????????????????????????????PPO???TD?????????

        with tf.GradientTape() as tape:
            v = self.critic(s)
            advantage = tfdc_r - v                  # ??????????????????td-error
            closs = tf.reduce_mean(tf.square(advantage))

        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

        return closs

    def cal_adv(self, tfs, tfdc_r):
        '''
        ??????advantage????????????td-error
        '''
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)           # advantage = r - gamma * V(s_)
        return advantage.numpy()

    def update(self, s, a, r):
        '''
        Update parameter with the constraint of KL divergent
        :param s: state
        :param a: act
        :param r: reward
        :return: None
        '''
        aloss = []
        closs = []
        s, a, r = s.astype(np.float32), a.astype(np.float32), r.astype(np.float32)

        self.update_old_pi()
        adv = self.cal_adv(s, r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # sometimes helpful

        # update actor
        #### PPO1????????????:
        # ?????????????????? adaptive KL penalty
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                kl = self.a_train(s, a, adv)
                if kl > 4 * METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(
                METHOD['lam'], 1e-4, 10
            )  # sometimes explode, this clipping is MorvanZhou's solution

        #### PPO2??????????????????????????????a_train??????:
        # clipping method, find this is better (OpenAI's paper)
        else:  
            for _ in range(A_UPDATE_STEPS):
                alossi = self.a_train(s, a, adv)
                aloss.append(alossi)

        # ?????? critic
        for _ in range(C_UPDATE_STEPS):
            clossi = self.c_train(r, s)
            closs.append(clossi)

        return aloss, closs

    def _build_anet(self, name, trainable):
        '''
        Build policy network
        :param name: name
        :param trainable: trainable flag
        :return: policy network
        '''
        # ??????????????????????????????mu???sigma???
        tfs = tl.layers.Input([None, S_DIM], tf.float32, name + '_state')
        l1 = tl.layers.Dense(100, tf.nn.sigmoid, name=name + '_l1')(tfs)

        a = tl.layers.Dense(A_DIM, tf.nn.tanh, name=name + '_a')(l1)
        mu = tl.layers.Lambda(lambda x: x * 2, name=name + '_lambda')(a)

        sigma = tl.layers.Dense(A_DIM, tf.nn.softplus, name=name + '_sigma')(l1)

        model = tl.models.Model(tfs, [mu, sigma], name)

        if trainable:
            model.train()
        else:
            model.eval()
        return model

    def choose_action(self, s):
        '''
        Choose action
        :param s: state
        :return: clipped act
        '''           
        s = s[np.newaxis, :].astype(np.float32) 
        mu, sigma = self.actor(s)                   # ??????actor??????????????????mu???sigma
        pi = tfp.distributions.Normal(mu, sigma)    # ???mu???sigma??????????????????
        a = tf.squeeze(pi.sample(1), axis=0)[0]     # ?????????????????????????????????
        return np.clip(a, -2, 2)                    # ??????sample???????????????????????????

    def get_v(self, s):
        '''
        ??????value??????
        '''
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]  # ??????????????????????????????
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_weights_to_hdf5('model/new_modle/ppo_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/new_modle/ppo_actor_old.hdf5', self.actor_old)
        tl.files.save_weights_to_hdf5('model/new_modle/ppo_critic.hdf5', self.critic)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/new_modle/ppo_actor.hdf5'.encode('utf-8').decode('utf-8'), self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/new_modle/ppo_actor_old.hdf5'.encode('utf-8').decode('utf-8'), self.actor_old)
        tl.files.load_hdf5_to_weights_in_order('model/new_modle/ppo_critic.hdf5'.encode('utf-8').decode('utf-8'), self.critic)

    def save_history(self, history, name):
        if not os.path.exists('./history/train/'):
            os.makedirs('./history/train/')
        name = os.path.join('history/train/', name)
        df = pd.DataFrame.from_dict(history)
        df.to_csv(name, index=False, encoding='utf-8')

if __name__ == '__main__':
    minloss = -1000
    interval_ = 0.001
    print("????????????")
    writer = SummaryWriter("runs/" + "1000/PPO")
    env = env.peg_in_hole_env()

    ppo = PPO()

    if args.train:
        all_ep_r = []
        updata, up_time, keep, keep_r, l, ep = 0, 0, 0, 0, 0, -1
        buffer_s, buffer_a, buffer_r = [], [], []
        buffer_s1, buffer_a1, buffer_r1 = [], [], []
        random_x, random_y, random_z = [], [], []
        # ???????????????
        while ep <= EP_MAX:
        # for ep in range(EP_MAX):
            ep += 1
            keep += 1
            s = env.reset()
            ep_r = 0
            t0 = time.time()
            for t in range(EP_LEN):  # in one episode
                a = ppo.choose_action(s)
                state, s_, r, done, interval = env.next_step(deepcopy(a), t, EP_LEN, interval_)

                if (done == 1 and t <= 2) or (t == 0 and state[2] < 0.6):
                    print("????????????##############################################")
                    keep = keep - 1
                    ep = ep - 1
                    s2 = env.Getdate()
                    pos = [-0.5, 0.2, s2[2], 0, -180, 0]
                    env.move_zp(pos)
                    break
                
                updata = updata + 1
                
                if t == EP_LEN - 1:
                    done = True
                    print("??????????????????????????????????????????????????????????????????????????????????????????")

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                s = s_
                ep_r = 0.9*ep_r + r

                force_torque['episode'].append(ep)
                force_torque['x_force'].append(s[6])
                force_torque['y_force'].append(s[7])
                force_torque['z_force'].append(s[8])
                force_torque['x_torque'].append(s[9])
                force_torque['y_torque'].append(s[10])
                force_torque['z_torque'].append(s[11])
                force_torque['reward'].append(r)
                force_torque['interval'].append(interval)

                tra['episode'].append(ep)
                tra['step'].append(t)
                tra['x'].append(state[0])
                tra['y'].append(state[1])
                tra['z'].append(state[2])
                tra['alph'].append(state[3])
                tra['beta'].append(state[4])
                tra['gama'].append(state[5])

                writer.add_scalar("interval", interval, updata)
                writer.add_scalar("r", r, updata)
                writer.add_scalar("x", a[0], updata)
                writer.add_scalar("y", a[1], updata)
                writer.add_scalar("z", a[2], updata)
                writer.add_scalar("alph", a[3], updata)
                writer.add_scalar("beta", a[4], updata)
                writer.add_scalar("gama", a[5], updata)
                writer.add_scalar("inter", a[6], updata)

                # N????????????????????????BATCH?????????????????????????????????
                if (ep % 2 == 0 and done and ep != 0) or updata % BATCH == 0:
                    print("???????????????????????????????????????????????????")
                    up_time += 1
                    v_s_ = ppo.get_v(s_)        # ??????n??????????????????state???v_s_

                    # ???PG??????????????????????????????
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    # ???????????????br????????????????????????reward??????????????????????????????V???
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    ppo.update(bs, ba, br)
                    aloss, closs = ppo.update(bs, ba, br)
                    writer.add_scalar("aloss", np.mean(aloss), up_time)
                    writer.add_scalar("closs", np.mean(closs), up_time)

                if done:
                    t1 = time.time() - t0
                    # data = json.dumps(str(1))
                    # data = data.encode('UTF-8')
                    # soc.sendall(data)
                    s2 = env.Getdate()
                    pos = [-0.5, 0.2, s2[2], 0, -180, 0]
                    env.move_zp(pos)
                    break
    
            if (done == 1 and t <= 2) or (t == 0 and state[2] < 0.6):
                continue
            
            if ep == 0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
            print(
                'Episode: {}/{}  | Episode Reward: {:.4f}  | Running step: {}  | Running Time: {:.4f}'.format(
                    ep, EP_MAX, ep_r, t,
                    t1
                )
            )
            history['episode'].append(ep)
            history['Episode_reward'].append(ep_r)
            history['step'].append(t)
            history['time'].append(t1)

    
            writer.add_scalar("Reward", ep_r, ep)
            writer.add_scalar("Step", t, ep)

            keep_r += ep_r

            if keep == 19:
                if keep_r > minloss:
                    l += 1
                    minloss = keep_r
                    ppo.save_ckpt()
                    print("keep", l, "times")
                keep = 0
                keep_r = 0

        plt.title('PPO')
        plt.plot(np.arange(len(all_ep_r)), all_ep_r)
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        plt.show()

        ppo.save_history(history, 'history.csv')
        ppo.save_history(force_torque, 'force_torque.csv')
        ppo.save_history(tra, 'tra.csv')
        ppo.save_history(buffer_s, 'state.csv')
        ppo.save_history(buffer_a, 'action.csv')
        ppo.save_history(buffer_r, 'reward.csv')