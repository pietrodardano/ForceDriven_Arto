import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

path = "/home/user/arto_ws/src/RoboDataML/datas/"

           #    0           1               2                   3                       4                       5                       6                       7           
columns1 = ["timestamp", "actual_q_4", "actual_TCP_force_0", "actual_TCP_force_1", "actual_TCP_force_2", "actual_TCP_force_3","actual_TCP_force_4","actual_TCP_force_5"]
           #    0           1           2         3           4           5           6      
columns2 = ["Timestamp", "Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"]
           #    0           1           2       3           4       5           6           7           8           9         10        11          12      
columns3 = ["Timestamp", "Pose_X", "Pose_Y", "Pose_Z", "Pose_Rx", "Pose_Ry", "Pose_Rz", "Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"]
labels   = ["Fx","Fy","Fz","Mx","My","Mz"]

def MAF(w):
    i=1
    a=0.40   #low "a" worse filtering but better tracking, if a >> the opposite
    wf = np.zeros(w.size)
    wf[0]=w[0]
    for i in range(w.size):
        wf[i]= a*w[i]+(1-a)*wf[i-1]             # LPF
        #wf[i] = (1-a)*w[i]+a*(w[i]+w[i-1])/2   # FIR 2° order
    #print(wf)
    return wf

def fplotter1(x, filecsv, columns):
    df = pd.read_csv(filecsv, usecols=columns)
    plt.rcParams["figure.figsize"] = [8.00, 4.50]
    plt.rcParams["figure.autolayout"] = True
    file_name = filecsv.split("/")[-1].split(".")[0]
    plt.title(f"Plot for {file_name}")
    j = 1
    min = 2
    for i in range(2, len(columns)):
        w = df[columns[i]].to_numpy()
        af = MAF(w)
        
        if(i>min+2): 
            j=2
        
        plt.subplot(2,1,j)
        plt.plot(df[x], df[columns[i]], label=labels[i-min])
        plt.plot(df[x], af, label= labels[i-min])
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def fplotter2(x, filecsv, columns):
    df = pd.read_csv(filecsv, usecols=columns)
    plt.rcParams["figure.figsize"] = [8.00, 4.50]
    plt.rcParams["figure.autolayout"] = True
    file_name = filecsv.split("/")[-1].split(".")[0]
    plt.title(f"Plot for {file_name}")
    j = 1
    min = 1
    for i in range(min, len(columns)):
        w = df[columns[i]].to_numpy()
        af = MAF(w)
        
        if(i>min+2): 
            j=2
        
        plt.subplot(2,1,j)
        plt.plot(df[x], df[columns[i]], label=labels[i-min])
        plt.plot(df[x], af, label= labels[i-min])
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def fplotter3(x, filecsv, columns):
    df = pd.read_csv(filecsv, usecols=columns)
    plt.rcParams["figure.figsize"] = [8.00, 4.50]
    plt.rcParams["figure.autolayout"] = True
    file_name = filecsv.split("/")[-1].split(".")[0]
    
    j = 1
    min = 7
    for i in range(min, len(columns)):
        w = df[columns[i]].to_numpy()
        af = MAF(w)
        
        if(i>min+2): 
            j=2
        
        plt.subplot(2,1,j)
        plt.plot(df[x], df[columns[i]], label=labels[i-min])
        plt.plot(df[x], af, label= labels[i-min])
        plt.legend()
    plt.suptitle(f"{file_name}")
    plt.tight_layout()
    plt.show()

def fplotterZ(forces, dur, columns):
    plt.rcParams["figure.figsize"] = [8.00, 4.50]
    plt.rcParams["figure.autolayout"] = True
    c = 1
    for fZ in forces:
        #filecsv = f"/home/user/arto_ws/src/RoboDataML/ARMpushtest/armbutt_{fZ}N_{dur}.csv"
        #filecsv = f"/home/user/arto_ws/src/RoboDataML/BlackButtPushTest/Butt_{fZ}N_{dur}.csv"
        #filecsv = f"/home/user/arto_ws/src/RoboDataML/WhiteRelePushtest/Butt_{fZ}N_{dur}.csv"
        #filecsv = f"/home/user/arto_ws/src/RoboDataML/YellowArrowPushTest/Butt_{fZ}N_{dur}.csv"
        filecsv = f"/home/user/arto_ws/src/RoboDataML/ALTPushTest/Butt_{fZ}N_{dur}.csv"
        #filecsv = f"/home/user/arto_ws/src/RoboDataML/FramePushTest/Butt_{fZ}N_{dur}.csv"
        #filecsv = f"/home/user/arto_ws/src/RoboDataML/BrakePushTest/Butt_{fZ}N_{dur}.csv"
        filecsv = "/home/user/arto_ws/src/RobotData_GRIPA320/TERR_but/Butt_8N_800_#78.csv"
        
        df = pd.read_csv(filecsv, usecols=columns)
        file_name = filecsv.split("/")[-1].split(".")[0]
        w = df[columns[9]].to_numpy()
        af = MAF(w)

        plt.subplot(len(forces),1,c)
        plt.plot(df[columns[0]], df[columns[9]], label=labels[2])
        #plt.plot(df[columns[0]], af, label=labels[2])
        plt.title(f"{file_name}")
        plt.legend()
        c+=1

    plt.tight_layout()
    plt.show()
        

def name(name):
    return path+name+".csv"

def main():

    # fplotter1(columns1[0], "/home/user/arto_ws/src/RoboDataML/datas/robot_data.csv", columns1)
    # fplotter1(columns1[0], "/home/user/arto_ws/src/RoboDataML/datas/w2360tcp.csv", columns1)
    # fplotter2(columns2[0], "/home/user/arto_ws/src/RoboDataML/datas/force_data1noreturn.csv", columns2)
    # fplotter2(columns2[0], "/home/user/arto_ws/src/RoboDataML/datas/force_dataCorrectPush.csv", columns2)
    # fplotter3(columns2[0], name("ButtonZ_4n"), columns3)
    # fplotter3(columns2[0], name("ButtonZ_3n"), columns3)
    # fplotter3(columns2[0], name("ButtonZ_2,5n"), columns3)
    # fplotter3(columns2[0], name("ButtonZ_2n"), columns3)
    # fplotter3(columns2[0], name("Pushmaiale_3n"), columns3)

    forces = [2,3,4,5]
    dur = [800, 1000, 1200]
    fplotterZ(forces, 1400, columns3)

    # fplotter3(columns2[0], "/home/user/arto_ws/src/RoboDataML/TestSim/Low_3state_lev_-5.csv", columns3)
    # fplotter3(columns2[0], "/home/user/arto_ws/src/RoboDataML/TestSim/To_lever_down_5.csv", columns3)
    # fplotter3(columns2[0], "/home/user/arto_ws/src/RoboDataML/TestSim/Brake_8.csv", columns3)
    # fplotter3(columns2[0], "/home/user/arto_ws/src/RoboDataML/ARMpushtest/armbutt_2N_1200.csv", columns3)

    return 0
    

if __name__ == "__main__":
    main()