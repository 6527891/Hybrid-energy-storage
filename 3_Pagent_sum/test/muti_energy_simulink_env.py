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

from rtwtypes import *

class MutiEnergy_simulink:
    def __init__(self, eptstep):
        csv_file_path = './pointlist_fina.csv'
        df = pd.read_csv(csv_file_path)
        self.point_P_wind = np.array(df['P_wind'])
        self.point_P_energy = np.array(df['P_energy'])
        self.point_sum = np.array(df['sum'])
        
        df_load = pd.read_csv('./10.csv', header=None)
        df_load.columns = ['timestamp','time','value','estimated','anomaly']
        df_load = df_load.iloc[1:]
        df_load['Index'] = range(len(df_load))
        self.P_load = np.float64(df_load['value'])[0:8640]/1800
        
        csv_file_path = './plant.c0.2021.texas.62417.csv'
        df = pd.read_csv(csv_file_path)
        
        ##########  Wind
        v_windin,v_windout,v_windn = 3,14,12
        Ttotal = 24*30
        Tbegin = 100
        
        Vwind_1h_yuan = df['ERA5 wind speed (m/s)']# ERA5 wind speed (m/s)  MERRA2 wind speed (m/s)
        self.Vwind_1h_yuan = np.array(Vwind_1h_yuan)[Tbegin:Tbegin+Ttotal]

        Pwind_1h_yuan = [0 if (vwind > v_windout or vwind < v_windin) 
                                  else 0.75*0.6*(vwind-v_windin)/(v_windn-v_windin) for vwind in self.Vwind_1h_yuan]    #4-14m/s
        
        original_data = np.array(Pwind_1h_yuan)
        time_points = np.linspace(0, Ttotal, num=Ttotal)
        interpolation_function = interp1d(time_points, original_data, kind='cubic')
        time_points_15min = np.linspace(0, Ttotal, num=Ttotal*4)    #15min
        Pwind_15min_yuan = interpolation_function(time_points_15min)
        time_points_5min = np.linspace(0, Ttotal, num=Ttotal*4*3)    #15min
        Pwind_5min_yuan = interpolation_function(time_points_5min)
        
        bool_array = Pwind_5min_yuan > 0.05
        np.random.seed(42)
        random_5_array = np.random.normal(loc=0, scale=0.03, size=Ttotal*4*3)
        Pwind_5min = np.multiply((Pwind_5min_yuan + random_5_array),bool_array)
        
        self.P_Wind = np.array(Pwind_5min) #+ 0.37
        self.P_Wind_5min = np.array(Pwind_5min_yuan) #+ 0.37
        self.P_Wind_yuan = Pwind_5min
        
        # simulink model
        if platform.system() == "Linux":
            self.dll = ctypes.cdll.LoadLibrary("./Simple_Droop.so")
        elif platform.system() == "Windows":
            self.dll = ctypes.cdll.LoadLibrary("./Simple_Droop_win64.dll")
        
        # Model entry point functions
        self.model_initialize = self.dll.Simple_Droop_initialize
        self.model_step = self.dll.Simple_Droop_step
        self.model_terminate = self.dll.Simple_Droop_terminate
        
        # Input Parameters
        self.Pset = real_T.in_dll(self.dll, "Pset")
        self.P_loadset = real_T.in_dll(self.dll, "P_loadset")
        self.P_Bset = real_T.in_dll(self.dll, "P_Bset")
        self.P_Hset = real_T.in_dll(self.dll, "P_Hset")
        self.P_SCset = real_T.in_dll(self.dll, "P_SCset")
        
        # Output Signals
        self.f_SM = real_T.in_dll(self.dll, "f_SM")       # Hz
        self.f_GFC = real_T.in_dll(self.dll, "f_GFC")
        self.P_SM = real_T.in_dll(self.dll, "P_SM")       # P_grid
        self.P_SUM = real_T.in_dll(self.dll, "P_SUM")
        self.P_Bs = real_T.in_dll(self.dll, "P_Bs")
        self.P_Hs = real_T.in_dll(self.dll, "P_Hs")
        self.P_SCs = real_T.in_dll(self.dll, "P_SCs")
        self.V_SM = real_T.in_dll(self.dll, "V_SM")       # V_grid
        self.V_GFC = real_T.in_dll(self.dll, "V_GFC")
        self.V_Bat = real_T.in_dll(self.dll, "V_Bat")     # Bat
        self.I_Bat = real_T.in_dll(self.dll, "I_Bat")
        self.SOC_Bat = real_T.in_dll(self.dll, "SOC_Bat")
        self.Vdc_Bat = real_T.in_dll(self.dll, "Vdc_Bat")
        self.SimTime = real_T.in_dll(self.dll, "SimTime") # time
        
        self.Pset.value = self.P_Wind[0]
        self.P_loadset.value = 0
        self.P_Bset.value = 0
        self.P_Hset.value = 0
        self.P_SCset.value = 0
        
        self.T_step = 60*5/eptstep
        self.H_Cap  = 2000000000*3600    #w*h  400mwh
        self.B_Cap  = 200000000*3600    #w*h  400mwh
        self.SC_Cap = 50000000*3600
        
        self.P_H = 0
        self.P_B = 0
        self.P_SC = 0
        self.H_SOC  = 0.50
        self.B_SOC  = 0.50
        self.SC_SOC = 0.50
        
        self.profit = 0
        self.count = [0,0,0]
        self.PHlist = []
        self.PBlist = []
        self.phsetlist = []
        self.pbsetlist = []
        
        # print(len(self.P_Wind))
        
    def step(self, ap, ah, ab, ctime): 
        self.profit = 0
        self.P_H = ah[0]/4        
        self.P_B = ab[0]/4
        self.P_SC = ap[0]/4 - self.P_H - self.P_B
        
        P_sum_r = 1
        for i in range(len(self.point_P_wind)):
            if self.point_P_wind[i] > self.P_Wind[ctime]-self.P_load[ctime]+0.5:
                P_sum_r = (self.point_sum[i]+self.point_sum[i-1])/2
                break
                
        
        if ctime == 0 or ctime == 1:
            self.H_SOC  = 0.50
            self.B_SOC  = 0.50
            self.SC_SOC = 0.50
        
        # model balance
        self.Pset.value = self.P_Wind[ctime]
        self.P_loadset.value = self.P_load[ctime]
        self.P_Bset.value = ab[0]/4
        self.P_Hset.value = ap[0]/4+0.5
        self.P_SCset.value = ap[0]/4 - self.P_H - self.P_B
        
        self.model_step()
        
        self.H_SOC -= self.P_H*self.T_step*100000000/self.H_Cap
        self.B_SOC -= self.P_B*self.T_step*100000000/self.B_Cap
        self.SC_SOC -= self.P_SC*self.T_step*100000000/self.SC_Cap
        
        # self.profit = 20/(np.abs( self.f_SM.value-50 )+0.01)
        self.profit = -10000*np.abs( self.f_SM.value-50)
        
        self.count = [1,1,1]
        self.Pdelt = 0
        Pdelneed = 0
        self.datall = np.array([self.f_SM.value, self.f_GFC.value, self.P_SM.value, self.P_H, self.P_B, self.P_SC, self.P_SUM.value,
                                self.P_load[ctime], self.P_Wind_yuan[ctime], self.V_SM.value, self.V_GFC.value, self.Vdc_Bat.value, 
                                self.H_SOC, self.B_SOC, self.SC_SOC])
        
        
        if (len(self.phsetlist)==0) or self.phsetlist[-1] != (self.P_load[ctime]-self.P_Wind_5min[ctime]) :
            self.phsetlist.append(self.P_load[ctime]-self.P_Wind_5min[ctime])
        if (len(self.pbsetlist)==0) or self.pbsetlist[-1] != (ap[0]/4 - self.P_H) :
            self.pbsetlist.append(ap[0]/4 - self.P_H)
        
        if len(self.pbsetlist) > 6:
            self.pbsetlist = self.pbsetlist[-10:]
        if len(self.phsetlist) > 12:
            self.phsetlist = self.phsetlist[-10:]
        
        
        Pstatefinal = np.array([self.Pset.value, (max(min(self.H_SOC,1),0)-0.5)])
        Hstatefinal = np.array([sum(self.phsetlist) / len(self.phsetlist), (max(min(self.B_SOC,1),0)-0.5)])
        Bstatefinal = np.array([sum(self.pbsetlist) / len(self.pbsetlist) ,  (max(min(self.SC_SOC,1),0)-0.5)])
        # print('ctime:',ctime,'||hlen:',len(self.phsetlist),'||blen:',len(self.pbsetlist))
        
        
        # Pstatefinal = np.array([self.Pset.value, (max(min(self.H_SOC,1),0)-0.5)])
        # Hstatefinal = np.array([self.P_load[ctime]-self.P_Wind_5min[ctime], (max(min(self.B_SOC,1),0)-0.5)])
        # Bstatefinal = np.array([(ap[0]/4 - self.P_H) ,  (max(min(self.SC_SOC,1),0)-0.5)])
        # Hstatefinal = np.array([P_sum_r-0.5-self.P_Wind_5min[ctime], (max(min(self.B_SOC,1),0)-0.5)])
        # Bstatefinal = np.array([(self.P_Wind_5min[ctime] + self.P_H - 0.5) ,  0]) #(max(min(self.SC_SOC,1),0)-0.5)
        statefinal = [Pstatefinal,Hstatefinal,Bstatefinal]
        
        return statefinal, self.profit, False, self.Pdelt, Pdelneed, self.count, self.datall, self.B_SOC, self.P_Wind


    def reset(self):
        self.model_terminate()
        
        self.Pset.value = self.P_Wind[0]
        self.P_loadset.value = 0
        self.P_Bset.value = 0
        self.P_Hset.value = 0
        self.P_SCset.value = 0
        
        self.P_B = 0
        self.P_H = 0
        self.P_SC = 0
        self.B_SOC  = 0.50
        self.SC_SOC  = 0.50
        self.H_SOC  = 0.50
        
        self.model_initialize()
        
        self.profit = 0
        self.count = [0,0,0]
        self.Pdelt = 0
        Pdelneed = 0
        self.PHlist = []
        self.PBlist = []
        
        self.phsetlist = []
        self.pbsetlist = []
        
        self.datall = np.array([self.f_SM.value, self.f_GFC.value, self.P_SM.value, self.P_H, self.P_B, self.P_SC, self.P_SUM.value,
                                self.P_load[0], self.P_Wind_yuan[0], self.V_SM.value, self.V_GFC.value, self.Vdc_Bat.value, 
                                self.H_SOC, self.B_SOC, self.SC_SOC])
        
        Pstatefinal = np.array([self.Pset.value, self.H_SOC-0.5])
        Hstatefinal = np.array([0.5-self.P_Wind_5min[0], (max(min(self.B_SOC,1),0)-0.5)])
        Bstatefinal = np.array([0 ,  (max(min(self.SC_SOC,1),0)-0.5)])
        statefinal_reset = [Pstatefinal,Hstatefinal,Bstatefinal]
        
        return statefinal_reset



    
    




##steptime=0.0001s       Tmax=20s       

if __name__ == "__main__":
    env = MutiEnergy_simulink(50000)
    
    for ep in range(1):
        env.reset()
        ap ,ah ,ab = [0.2],[0.2],[0.1]
        aset = 0.75
        socblist,socsclist,P_sm_list, P_sum_list,rlist,flist,f1l = [],[],[],[],[],[],[]
        
        for t in range(int(300000 + 10000*5)):
            
            if t<500000:
                ctime = 0
            else:
                ctime = (t-300000)/150000
            
            
            # ctime = int(t/200000)
            s,r,_,pdel,Pdelneed,count,datall,soc_b,pwind = env.step(ap, ah, ab, int(ctime))
            
            if t%150000 == 0:
                # print(datall[2])
                print(env.Pset.value,soc_b)
            
            P_sm_list.append(datall[2])
            P_sum_list.append(datall[5])
            flist.append(datall[0])
            socblist.append(soc_b)
            rlist.append(r)
            
        
    
        plt.plot(P_sm_list,label='sm')
        plt.plot(P_sum_list,label='energy')
        plt.legend()
        plt.savefig('./P.png')
        plt.close()
        
        plt.plot(pwind)
        plt.plot(env.P_load)
        plt.legend()
        plt.savefig('./Pwindload.png')
        plt.close()
        
        pw_pl = pwind - env.P_load + 0.5
        plt.plot(pw_pl)
        plt.savefig('./pw_pl.png')
        plt.close()
        print('min:',min(pw_pl),'max:',max(pw_pl),'mean:',np.mean(pw_pl))
        
        plt.plot(env.Vwind_1h_yuan)
        plt.legend()
        plt.savefig('./vwind.png')
        plt.close()
        
        plt.plot(socblist,label='bat')
        plt.plot(socsclist,label='sc')
        plt.legend()
        plt.savefig('./soc.png')
        plt.close()
        
        flist50 = [i-50 for i in flist]
        plt.plot(flist50[50000:],label='SM_f')
        plt.legend()
        plt.savefig('./f.png')
        plt.close()
        
        plt.plot(rlist)
        plt.savefig('./r.png')
        plt.close()


