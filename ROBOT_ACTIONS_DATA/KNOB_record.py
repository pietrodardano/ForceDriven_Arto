import os
import csv
import time
import timeit
import pandas as pd
import random
import sys
sys.path.append('/home/user/thesis_ws/src/ROBOT_COMMAND_CODE')

import rtde_receive
import rtde_control

from robotiq_gripper import RobotiqGripper

robot_ip = "192.168.1.102"
# RTDE communication parameters
rtde_c = rtde_control.RTDEControlInterface(robot_ip)
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
gripper = RobotiqGripper()
gripper.connect(robot_ip, 63352)

#### SIM320 POINTS
from POINTS_A320 import *
# MOVE PARAMETERS
speed    = 1.2
accel    = 0.8

# GET INPUT FROM TERMINAL 
def get_user_input():
    while True:
        try:
            user_input = int(input("0 +++ 1: 1-5 +++ 2: 2-4 +++ 3 :3-6 -->  "))
            if 0 <= user_input <= 6:
                return user_input
            else:
                user_input = int(input("0 +++ 1: 1-5 +++ 2: 2-4 +++ 3 :3-6 -->  "))
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def get_user_label():
    user_input = input("Was it CORRECT? Y or N: ").upper()
    
    while user_input not in ['Y', 'N']:
        print("Invalid input. Please enter Y or N.")
        user_input = input("Was it CORRECT? Y or N: ").upper()
    
    result = {'N': 0, 'Y': 1}.get(user_input, None) # to binary
    return result


def count_files(folder_path):
    # Initialize a counter for files
    file_count = 0
    
    # Iterate over all items in the directory
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # Check if the item is a file
        if os.path.isfile(item_path):
            file_count += 1
    
    return file_count

def rotateCW(fZ, dur, GOAL, APPROACH, FOLDER_PATH):
    if fZ >0:
        return doKnob(fZ, dur, GOAL, APPROACH, FOLDER_PATH)
    elif fZ<0:
        return doKnob(-fZ, dur, GOAL, APPROACH, FOLDER_PATH)

def rotateCCW(fZ, dur, GOAL, APPROACH, FOLDER_PATH):
    if fZ <0:
        return doKnob(fZ, dur, GOAL, APPROACH, FOLDER_PATH)
    elif fZ>0:
        return doKnob(-fZ, dur, GOAL, APPROACH, FOLDER_PATH)

def doKnob(fZ, dur, GOAL, APPROACH, FOLDER_PATH):

    # FORCE SETTINGS
    selection_vector    = [0, 0,  0, 0, 0, 1]
    dir_force           = [0, 0,  0, 0, 0, fZ]
    force_type          = 2
    limits              = [0.02, 0.02, 1, 0.01, 0.01, 4]
    
    cnt_files = count_files(FOLDER_PATH)
    file_name = f"KNOB_CLAMPthenZERO_REAL_OPEN196_{fZ}N_{dur}_#{cnt_files}.csv"
    csv_file_path = os.path.join(FOLDER_PATH, file_name)
    
    print("pinting in", csv_file_path)
    # Open CSV file for writing
    
    
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ["Timestamp", "Pose_X", "Pose_Y", "Pose_Z", "Pose_Rx", "Pose_Ry", "Pose_Rz", "Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z", "Q_w1", "Q_w2", "Q_w3"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write CSV header
        writer.writeheader()

        # FIRST MOVES 
        rtde_c.moveJ_IK(GOAL, speed, accel)

        
        time.sleep(.1)
        gripper.move_and_wait_for_pos(196, 90, 0)
        time.sleep(.1)
        
        # Record data for a specified duration (e.g., 10 seconds)
        start_time = time.time()
        target_frequency = 500  # Hz
        i = 0

        #  ZEROING FTSENSOR
        rtde_c.zeroFtSensor()

        print("Registro da ora: ", start_time)
        while (i<dur):
            # Record start time for this iteration
            iteration_start_time = timeit.default_timer()

            # Execute 500Hz control loop for 4 seconds, each cycle is 2ms
            t_start = rtde_c.initPeriod()
            
            task_frame = rtde_r.getActualTCPPose()  #is updated time by time
            rtde_c.forceMode(task_frame, selection_vector, dir_force, force_type, limits)
            rtde_c.waitPeriod(t_start)

            # Read force data
            force_data = rtde_r.getActualTCPForce()
            pose_data  = rtde_r.getActualTCPPose()
            q_data = rtde_r.getActualQ()

            timestamp = time.time() - start_time

            # Write data to CSV file
            writer.writerow({
                "Timestamp": timestamp,
                "Pose_X" : pose_data[0],
                "Pose_Y" : pose_data[1],
                "Pose_Z" : pose_data[2], 
                "Pose_Rx": pose_data[3],
                "Pose_Ry": pose_data[4],
                "Pose_Rz": pose_data[5],
                "Force_X": force_data[0],
                "Force_Y": force_data[1],
                "Force_Z": force_data[2],
                "Torque_X": force_data[3],
                "Torque_Y": force_data[4],
                "Torque_Z": force_data[5],
                "Q_w1":     q_data[3],
                "Q_w2":     q_data[4],
                "Q_w3":     q_data[5]
            })

            # Calculate time taken for this iteration
            iteration_time = timeit.default_timer() - iteration_start_time

            # Adjust sleep duration for target frequency
            sleep_duration = 1.0 / target_frequency - iteration_time
            if sleep_duration > 0: print("sleep?")
                #time.sleep(sleep_duration)
            i += 1
        rtde_c.forceModeStop()
        print("finished")

        time.sleep(0.8)
        gripper.move_and_wait_for_pos(130, 90, 0)
        time.sleep(0.2)
    rtde_c.moveJ(APPROACH, speed, accel)
    
    # Prompt user for input
    user_input = get_user_input()
    #user_input = 0.0
    
    # Open the existing CSV file and read its contents
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    data[0].append('Y')
    data[1].append(user_input)
    
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
    return csv_file_path

def main():

    # Connect to the robot
    rtde_c.disconnect()
    rtde_r.disconnect()
    rtde_r.reconnect()
    rtde_c.reconnect()

    payl = rtde_r.getPayload()
    print(payl)
    rtde_c.setTcp([0, 0, 0.165, 0, 0, 0])
    print(rtde_c.getTCPOffset())
        
    FOLDER_PATH1 = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/KNOB_OPEN/FCU/"
    FOLDER_PATH2 = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/KNOB_OPEN/ECAM/"

    # JOINT POSE POINTS _q
    #APPROACH= [-1.525883, -0.998204, 1.027922, 3.928726, -0.210365, -0.253006] # FCU dx
    APPROACH = [-1.397082, -1.800153, 1.821568, 4.108171, -0.398757, -1.055058] # FCU sx
    #GOAL1    = [-0.552308, 0.670132, 0.51113, -1.440935, -1.431635, 0.950753] #  FCU DX
    #GOAL1 = [-0.30111, 0.585286, 0.465862, -2.022119, -0.400974, 0.213766]  # Away Dx
    #GOAL1   = [-0.321963, 0.639734, 0.404281, -2.143322, -0.364276, 0.326397] # Away SX
    #GOAL1  = [-0.555412, 0.357536, 0.508045, -0.630039, -1.823052, 0.428344] # Sx 20
    #GOAL1 = [-0.554406, 0.355258, 0.508495, -1.459371, -1.497923, 1.053156]  # Sx 40
    #GOAL1 = [-0.552726, 0.357923, 0.50800, -1.53912, -1.443787, 1.020017] # SX 40 BETTER
    #GOAL1 = [-0.55445, 0.359547, 0.510068, -2.061138, -0.85659, 1.485008] # SX 80
    GOAL1   = [-0.553981, 0.357699, 0.508584, -2.138285, -0.781205, 1.495215] # SX 80 BETTER
    #GOAL1 = [-0.554219, 0.355487, 0.507843, 2.538242, 0.017636, -1.801958] # SX 160
    #GOAL1 = [-0.554043, 0.356309, 0.508342, 2.128633, -0.90935, -1.450675]
    #APPROACH = [-0.980204, -0.824352, 1.202437, 4.143493, -1.395153, -0.956457] # ECAM
    #GOAL1 = [-0.593425, 0.542964, -0.012062, -2.14914, -2.154136, 0.239314] # ECAM EIF DMC NORM
    #GOAL1 = [-0.592002, 0.542486, -0.012024, -0.851771, -2.904899, 0.128759] # ECAM EIF DMC CAPT
    #GOAL1 = [-0.554029, 0.717802, 0.513484, -2.531848, 0.09777, 1.826729] # NAV knob DX PLAN
    GOAL2 = [-0.591962, 0.542524, -0.011619, 3.006197, 0.838261, -0.217715] # ECAM EIF DMC F/0
    TWO = 0
    
    #rtde_c.moveJ(HOME_q) 

    # FPR 50 VAL MEAS
    #APPROACH = [-1.461045, -1.18967, 1.142448, 4.473905, -0.277067, -0.309503]  # APPR FCU Metric Alt knob
    #GOAL1 = [-0.545578, 0.608367, 0.515538, -1.569004, -1.387977, 1.052602]  # Metric alt knob
    
    
    rtde_c.moveJ(APPROACH, speed, accel)
    gripper.move_and_wait_for_pos(120, 90, 30)
    Zvalue = [3,3,3,3]

    if(1):
        for fZ in Zvalue:
            print('---- now ', fZ, 'N ----')

            # 3N 1600 is 1 N

            for dur in range(2300, 2400, 100):
                print('GOAL1 --> force:', fZ, '   dur:', dur)
                result_file_path = rotateCCW(fZ, dur, GOAL1, APPROACH, FOLDER_PATH1)
                print('')
                
                data = pd.read_csv(result_file_path)
                force_z_values = data['Torque_Z'].unique()
                #rtde_c.forceModeSetGainScaling(0.1)

                # Check if all force_z values are equal
                if len(force_z_values) == 1:
                    file_name = os.path.basename(result_file_path)
                    print(f"Deleting file: {file_name}")
                    os.remove(result_file_path)
                    break

                if GOAL1 != GOAL2 and TWO == 1:
                    print('GOAL2 --> force:', fZ, '   dur:', dur)
                    result_file_path = rotateCCW(fZ, dur, GOAL2, APPROACH, FOLDER_PATH2)
                    print('')
                    data = pd.read_csv(result_file_path)
                    force_z_values = data['Torque_Z'].unique()
                
                    # Check if all force_z values are equal
                    if len(force_z_values) == 1:
                        file_name = os.path.basename(result_file_path)
                        print(f"Deleting file: {file_name}")
                        os.remove(result_file_path)
                        break
                time.sleep(1.2)
    
    else:
        for i in range(1):
            doKnob(8, 800, GOAL1, APPROACH, FOLDER_PATH1)
            c = input('gnegne --- ')
            time.sleep(0.8)

    # Disconnect from the robot
    rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()
    

if __name__ == "__main__":
    main()