import time
from muti_energy_env import MutiEnergy
import pandas as pd
import csv

import os
import platform

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
# tf.compat.v1.Session()
tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()


EP_MAX = 70000
EP_LEN = 499
GAMMA = 0.9
A_LR = 0.000001
C_LR = 0.000002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 2, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

if platform.system() == "Linux":
    EP_figure = 5000     #every EP_figure plot fig
elif platform.system() == "Windows":
    EP_figure = 1
    EP_MAX = 1

class BPPO(object):
    def __init__(self):
        S_DIM, A_DIM = 2, 1
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('Bcritic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('Bpi', trainable=True)
        oldpi, oldpi_params = self._build_anet('Holdpi', trainable=False)
        with tf.variable_scope('Bsample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('Bupdate_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('Bloss'):
            with tf.variable_scope('Bsurrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        # tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
    
    def save_model(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
        print("Model saved at", save_path)
        
    def load_model(self, load_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)
        print("Model loaded from", load_path)


env = MutiEnergy()
bppo = BPPO()
all_ep_r = []
rlist = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    
    pwindlist, pblist, phlist, pdeltlist, socblist, sochlist = [],[],[],[],[],[]
    for t in range(EP_LEN):    # in one episode
    
        a = bppo.choose_action(s)
        # s_, r, done, _ = env.step(a)
        s_, r, done, count, Pall, soch, pw = env.step(a,t)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r))    # normalize reward, find to be useful
        
        print('sar',s,a,r, 'cr',-s[0]+a[0]/4 )
        
        s = s_
        ep_r += r
        
        rlist.append(r)
        pwindlist.append(Pall[0])
        phlist.append(Pall[1])
        sochlist.append(Pall[2])
        pblist.append(Pall[3])
        socblist.append(Pall[4])
        pdeltlist.append(Pall[-1])
        
        
        
        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = bppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            bppo.update(bs, ba, br)
    # if ep == 0: all_ep_r.append(ep_r)
    # else: all_ep_r.append(ep_r)
    all_ep_r.append(ep_r)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %.2f" % ep_r,
        "|Pdel: %.4f" % np.mean(pdeltlist),
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else ''
    )
    print(soch[0:5])
    print(0.5-pw[0:5])
    
    if (ep+1) % EP_figure == 0:

        plt.plot(np.array(all_ep_r))
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        plt.savefig('./ep_r.png')
        plt.close()
        
        plt.plot(np.array(pdeltlist))
        plt.savefig('./Pdelt.png')
        plt.close()
        
        plt.plot(np.array(pwindlist), '-',color='r', label='Pwind')
        plt.plot(np.array(phlist), '--',color='black', label='Ph')
        plt.plot(np.array(pblist), '--',color='blue', label='Pb')
        plt.legend()
        plt.savefig('./Pneed.png')
        plt.close()
        
        plt.plot(np.array(socblist), '-',color='g', label='socb')
        plt.plot(np.array(sochlist), '-',color='r', label='soch')
        plt.legend()
        plt.savefig('./soc.png')
        plt.close()
        
        data = list(zip(pblist,pwindlist,pdeltlist,socblist))
        csv_file_path = "dataPout.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
            
        data = list(zip(all_ep_r))
        csv_file_path = "datarhout.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
            
        # pppo.save_model('pmodel.ckpt')
        bppo.save_model('hmodel.ckpt')
        
        
        # test
        pdelt = -np.linspace(start=0, stop=0.6, num=20)+0.21
        soch = np.linspace(start=0.1, stop=0.9, num=20)
        phlist = []

        for ep in range(len(pdelt)):
            a = bppo.choose_action(np.array([pdelt[ep],0]))
            phlist.append(a[0]/4)

        plt.plot(np.array(phlist), '-',color='g', label='aph')
        plt.plot(np.array(pdelt+0.5), '-',color='r', label='pdelt')
        plt.legend()
        plt.savefig('./pjieguo.png')
        plt.close()

        phlist = []
        for ep in range(len(soch)):
            a = bppo.choose_action(np.array([0,soch[ep]-0.5]))
            phlist.append(a[0]/4)

        plt.plot(np.array(phlist), '-',color='g', label='aph')
        plt.plot(np.array(soch), '-',color='r', label='soch')
        plt.legend()
        plt.savefig('./socjieguo.png')
        plt.close()
        



print('A_LR:%f' %A_LR, 'C_LR:%f' %C_LR, 'BATCH:%i' %BATCH)


