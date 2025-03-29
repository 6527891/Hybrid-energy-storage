import time
from envs.muti_energy_simulink_env import MutiEnergy_simulink
from algorithms.ppo3 import *
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

eptstep = 50000
EP_MAX = 500
EP_LEN = 300000 + eptstep*12*24*30


train_test=0 
if platform.system() == "Linux":
    EP_figure = 1     #every EP_figure plot fig
elif platform.system() == "Windows":
    EP_figure = 1
    EP_MAX = 1



env = MutiEnergy_simulink(eptstep)


hppo = HPPO()
tf.reset_default_graph()
bppo = BPPO()
tf.reset_default_graph()
scppo = PPPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    hbuffer_s, hbuffer_a, hbuffer_r, bbuffer_s, bbuffer_a, bbuffer_r,scbuffer_s, scbuffer_a, scbuffer_r = [], [], [],[], [], [],[], [], []
    
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
            asc = scppo.choose_action(s[2])       #5min
            if ctime % 3 == 0:
                ab = bppo.choose_action(s[1])   #15min
            if ctime % 12 == 0:
                ah = hppo.choose_action(s[0])   #1h
            # print(t,ctime,'p',s[0],ap,'h',s[1],ah,'b',s[2],ab)
        
        s_, rstep, done, pdel, Pdelneed, count, datall, soc_b, pwind = env.step(ah,ab,asc,ctime)
        r = rstep
        
        if t == 300000 + eptstep -1 + eptstep*ctime :
            
            scbuffer_s.append(s[2])
            scbuffer_a.append(asc)
            scbuffer_r.append((r[2]))    # normalize reward, find to be useful
            if ctime % 3 == 0:
                bbuffer_s.append(s[1])
                bbuffer_a.append(ab)
                bbuffer_r.append((r[1]))    # normalize reward, find to be useful
            if ctime % 12 == 0:
                hbuffer_s.append(s[0])
                hbuffer_a.append(ah)
                hbuffer_r.append((r[0]))    # normalize reward, find to be useful
            
            
            # update hppo
            if (len(hbuffer_s)) % BATCH == 0 or t == EP_LEN-1:
                v_s_ = hppo.get_v(s_[1])
                discounted_r = []
                for ri in hbuffer_r[::-1]:
                    v_s_ = ri + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
        
                bs, ba, br = np.vstack(hbuffer_s), np.vstack(hbuffer_a), np.array(discounted_r)[:, np.newaxis]
                hbuffer_s, hbuffer_a, hbuffer_r = [], [], []
                hppo.update(bs, ba, br)
                
            # update dcppo
            if (len(bbuffer_s)) % BATCH == 0 or t == EP_LEN-1:
                v_s_ = bppo.get_v(s_[2])
                discounted_r = []
                for ri in bbuffer_r[::-1]:
                    v_s_ = ri + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
        
                bs, ba, br = np.vstack(bbuffer_s), np.vstack(bbuffer_a), np.array(discounted_r)[:, np.newaxis]
                bbuffer_s, bbuffer_a, bbuffer_r = [], [], []
                bppo.update(bs, ba, br)
            
            # update dcppo
            if (len(scbuffer_s)) % BATCH == 0 or t == EP_LEN-1:
                v_s_ = scppo.get_v(s_[0])
                discounted_r = []
                for ri in scbuffer_r[::-1]:
                    v_s_ = ri + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
        
                bs, ba, br = np.vstack(scbuffer_s), np.vstack(scbuffer_a), np.array(discounted_r)[:, np.newaxis]
                scbuffer_s, scbuffer_a, scbuffer_r = [], [], []
                scppo.update(bs, ba, br)
            
            
            
            
            rlist.append(r)
            print(ep,ctime,'|p:',s[0],asc[0]/4+0.5,'|b:',s[1],ab[0]/4+0.5,r)
            
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
        
        

# hppo.save_model('hmodel.ckpt')
# bppo.save_model('bmodel.ckpt')
# scppo.save_model('scmodel.ckpt')
print('A_LR:%f' %A_LR, 'C_LR:%f' %C_LR, 'BATCH:%i' %BATCH)

