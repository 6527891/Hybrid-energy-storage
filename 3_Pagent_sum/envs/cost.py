import ctypes
import platform

import numpy as np
from os import path
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv


class Cost_sum:
    def __init__(self):
        self.wf = 2000
        self.wsc = 65
        self.wb = 50
        self.wh = 58
        
        ## b
        self.wb1 = 30
        self.costb = 0.2
        self.costbdegr = 0
        
        ## h
        self.wh1 = 30
        self.costhelzon = 0.01
        self.costhelzup = 0.126
        self.costhelzdown = 0.062
        self.costhfcon = 0.01
        self.costhfcup = 0.126
        self.costhfcdown = 0.062
        
    def Fcost(self, f): 
        cost_f = self.wf*(f-50)*(f-50)
        
        return cost_f
    
    def SCcost(self, socsc): 
        cost_sc = self.wsc*(socsc-0.5)*(socsc-0.5)
        
        return cost_sc

    def Bcost(self, socb, Pb): 
        cost_bsoc = self.wb1*(socb-0.5)*(socb-0.5)
        cost_bP = self.costb*np.abs(Pb) + self.costbdegr*np.abs(Pb)*np.abs(Pb)
        cost_b = self.wb*(cost_bsoc+cost_bP)
        
        return cost_b
    
    def Hcost(self, soch, hstat, Ph): 
        cost_hsoc = self.wh1*(soch-0.5)*(soch-0.5)
        
        if Ph>0:
            cost_on = self.costhfcon*Ph
        elif Ph<0:
            cost_on = -self.costhelzon*Ph
        if hstat == 1:
            cost_stat = self.costhelzup
        elif hstat == 0:
            cost_stat = self.costhelzdown
        else:
            cost_stat = 0
            

        cost_h = self.wh*(cost_hsoc + cost_on + cost_stat)
        
        return cost_h