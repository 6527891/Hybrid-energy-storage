import time
from muti_energy_simulink_env import MutiEnergy_simulink
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

eptstep = 50000#50000
EP_MAX = 1
EP_LEN = 300000 + eptstep*12*24*30
# EP_LEN = 300000 + 200000*10
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 1, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization



if platform.system() == "Linux":
    EP_figure = 1     #every EP_figure plot fig
elif platform.system() == "Windows":
    EP_figure = 1
    EP_MAX = 1



class PPPO(object):
    def __init__(self):
        S_DIM, A_DIM = 2, 1
        
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('Pcritic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('Ppi', trainable=True)
        oldpi, oldpi_params = self._build_anet('Poldpi', trainable=False)
        with tf.variable_scope('Psample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('Pupdate_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('Ploss'):
            with tf.variable_scope('Psurrogate'):
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

class HPPO(object):
    def __init__(self):
        S_DIM, A_DIM = 2, 1
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('Hcritic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('Hpi', trainable=True)
        oldpi, oldpi_params = self._build_anet('Boldpi', trainable=False)
        with tf.variable_scope('Hsample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('Hupdate_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('Hloss'):
            with tf.variable_scope('Hsurrogate'):
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


env = MutiEnergy_simulink(eptstep)
pppo = PPPO()
pppo.load_model('./model/pmodel/hmodel.ckpt')
tf.reset_default_graph()
hppo = HPPO()
hppo.load_model('./model/hmodel/hmodel.ckpt')
tf.reset_default_graph()
bppo = BPPO()
bppo.load_model('./model/bmodel/hmodel.ckpt')   #保存的模型命名错误

all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    
    thstep = 0
    ep_r = 0
    batch_t = 0
    sochlist,socblist,socsclist,rlist,flist = [],[],[],[],[]
    phlist,pblist,psclist,psmlist,psumlist,ploadlist = [],[],[],[],[],[]
    countlist = [0,0,0]
    
    for t in range(EP_LEN):    # in one episode        300000 + 200000*60*24
        if t<300000 + eptstep :
            ctime = 0
        else:
            ctime = int((t-300000)/eptstep)
            
        if t==0 or t==300000 + eptstep*ctime + 1:
            ap = pppo.choose_action(s[0])       #5min
            if ctime % 3 == 0:
                ab = bppo.choose_action(s[2])   #15min
            if ctime % 12 == 0:
                ah = hppo.choose_action(s[1])   #1h
            print(t,ctime,'p',s[0],ap,'h',s[1],ah,'b',s[2],ab)
        
        s_, rstep, done, pdel, Pdelneed, count, datall, soc_b, pwind = env.step(ap,ah,ab,ctime)
        r = rstep
        
        if t == 300000 + eptstep -1 + eptstep*ctime :
                
            rlist.append(r)
            print(ep,ctime,'|p:',s[0],ap[0]/4+0.5,'|b:',s[1],ab[0]/4+0.5,r)
            
            phlist.append(datall[3])
            pblist.append(datall[4])
            psclist.append(datall[5])
            psmlist.append(datall[2])
            psumlist.append(datall[6])
            ploadlist.append(datall[7])
            flist.append(datall[0])
            sochlist.append(datall[-3])
            socblist.append(datall[-2])
            socsclist.append(datall[-1])
            all_ep_r.append(r)
            
        s = s_
    
        # psmlist.append(datall[2])
        # pblist.append(datall[3])
        # psclist.append(datall[4])
        # psumlist.append(datall[5])
        # flist.append(datall[0])
        # socblist.append(soc_b)
        # socsclist.append(soc_sc)
        # all_ep_r.append(r)
        
    if (ep+1)%EP_figure == 0:
        
        plt.plot(np.array(all_ep_r))
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        plt.savefig('./ep_rb.png')
        plt.close()
        
        plt.plot(np.array(psclist), '-',color='g', label='Psc') 
        plt.plot(np.array(pblist), '-',color='b', label='Pb')
        plt.plot(np.array(phlist), '-',color='r', label='Ph')
        plt.legend()
        plt.savefig('./Phbsc.png')
        plt.close()
        
        penergy = np.array(psclist) + np.array(pblist) + np.array(phlist)
        plt.plot(np.array(ploadlist), '-',color='g', label='Pload') 
        plt.plot(np.array(pwind), '-',color='b', label='Pwind')
        plt.plot(np.array(penergy), '-',color='r', label='Penergy')
        plt.legend()
        plt.savefig('./Pwle.png')
        plt.close()
        
        plt.plot(np.array(psmlist), '-',color='g', label='Psm')
        plt.plot(np.array(psumlist), '--',color='blue', label='Psum')
        plt.legend()
        plt.savefig('./Penergy_sm.png')
        plt.close()
        
        plt.plot(np.array(sochlist), '-',color='r', label='Hsoc')
        plt.plot(np.array(socblist), '-',color='b', label='Bsoc')
        plt.plot(np.array(socsclist), '-',color='g', label='SCsoc')
        plt.legend()
        plt.savefig('./soc.png')
        plt.close()
        
        plt.plot(np.array(flist), '-',color='g', label='fsm')
        plt.legend()
        plt.savefig('./f.png')
        plt.close()

        data = list(zip(psmlist,phlist,pblist,psclist,psumlist,sochlist,socblist,socsclist,flist))
        csv_file_path = "dataPout.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
            
        data = list(zip(rlist))
        csv_file_path = "datarpout.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        
        
print('A_LR:%f' %A_LR, 'C_LR:%f' %C_LR, 'BATCH:%i' %BATCH)



