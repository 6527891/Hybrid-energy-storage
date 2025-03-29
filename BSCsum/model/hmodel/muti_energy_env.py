import numpy as np
from os import path
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv

import random

# actor_dim = 1
# state_dim = 4
class MutiEnergy:
    def __init__(self):
        
        Pwind_1h_yuan = []
        self.B_SOC_random = []
        random.seed(42)
        for i in range(500):
            Pwind_1h_yuan.append(random.random()*0.6)
            self.B_SOC_random.append(random.random()*0.8+0.1)
        
        self.P_Wind = np.array(Pwind_1h_yuan) + 0.29
        self.P_Wind_yuan = Pwind_1h_yuan
        
        self.P_B = 0
        self.P_H = 0
        self.P_SC = 0
        self.B_SOC  = 0.50
        self.SC_SOC  = 0.50
        self.H_SOC  = 0.50
        
        self.profit = 0
        
        self.T_step = 60*60    #s
        
        self.B_Cap  = 500000000*3600    #ws=wh*3600   300mwh
        self.SC_Cap = 400000000*3600
        self.H_Cap  = 1000000000*3600
        
        self.count = [0,0,0]
        
        
    def step(self, ah , ctime): 
        self.profit = 0
        self.P_H = ah[0]/4
        self.P_B = 0.5 - ah[0]/4 - (self.P_Wind[ctime])
        
        self.H_SOC -= self.P_H*self.T_step*100000000/self.H_Cap
        self.B_SOC -= self.P_B*self.T_step*100000000/self.B_Cap
        
        self.P_delt = self.P_Wind[ctime] + self.P_H - 0.5
        self.Pall = np.array( [self.P_Wind_yuan[ctime], self.P_H, self.H_SOC, self.P_B, self.B_SOC, self.P_delt] )
        self.profit -= np.abs(self.P_Wind[ctime] + self.P_H - 0.5 + (self.B_SOC_random[ctime]-0.5)*1)
        
        statefinal = np.array([(0.5-self.P_Wind[ctime+1]), self.B_SOC_random[ctime+1]-0.5])
        
        return statefinal, self.profit, False, self.count, self.Pall, self.B_SOC_random, self.P_Wind
    
    def reset(self):
        self.P_B = 0
        self.P_H = 0
        self.P_SC = 0
        self.B_SOC  = 0.50
        self.SC_SOC  = 0.50
        self.H_SOC  = 0.50
        
        self.profit = 0
        self.count = [0,0,0]
        
        statefinal_reset = np.array([self.P_Wind_yuan[0], (self.B_SOC-0.5)])
        return statefinal_reset
    

    
if __name__ == "__main__":
    env = MutiEnergy()
    
    soclist,Pwindlist,Pblist = [],[],[]
    for ep in range(96):
        s,r,_,count,Pall,bsoc = env.step([1], ep)
        
        Pwindlist.append(Pall[0])
        Pblist.append(Pall[1])
        soclist.append(bsoc)
        
        # print(soc,env.P_Battery,env.P_B/env.B_Cap*100)
        
    plt.plot(Pwindlist)
    plt.plot(Pblist)
    plt.show()
    
    plt.plot(soclist)
    plt.show()