import os
import csv
import time
import timeit
import random
import pandas as pd
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
    '''
    0 --> Fail PULLING          # NOT ENOUGH
    1 --> Fail MOVING           # NOT ENOUGH OR WRONG DIRECTION!!
    2 --> SUCCESS
    3 -->  NOT USED (2--> HARD RELEASE HAPPENS TOO RARELY, not a problem tho)

    IF 2 AND 3 WILL BE MISSCLASSIFIED, I WILL JOIN THEM
    '''
    user_input = int(input("FAIL 0,1,2 or SUCCESS 3? +++ "))
    
    while user_input not in [0,1,2,3]:
        print("Invalid input. 0,1,2,3")
        user_input = int(input(" +++ 0, 1, 2, 3 +++ "))
    
    result = user_input
    return result


def get_user_COMMENT():
    user_input = input("NOT SUCCESS: no_left, no_right, missed_").upper()
    
    while user_input not in ['Y', 'N']:
        print("Invalid input. Please enter Y or N.")
        user_input = input("Was it successfull? Y or N: ").upper()
    
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

################################################################

def modify_values(values):
    #random.seed(42)
    modified_values = []
    for value in values:
        # Check if the value needs to be modified
        if value and False: # in [-3, 4, -4, -25, -28, 24, -24]
            change = random.randint(-8, 4)
            modified_value = value + change
        else:
            modified_value = value
        # Append the modified value to the list
        modified_values.append(modified_value)
    return modified_values


def FLAP_plus(dur, GOAL, APPROACH, FOLDER_PATH):

    # FORCE SETTINGS
    selection_vector    = [1,  0,  1,  0, 1, 0]
    dir_pullZ           = [4,  0, -15, 0, 0, 0]
    dir_Xincreas        = [15, 0, -15, 0, 0, 0]
    dir_finalYZ         = [4,  0,  -3,  0, 0, 0]
    force_type          = 2
    limits              = [8, 0.01, 8, 0.1, 0.1, 0.2]

    # Modify values
    modified_dir_pullZ = modify_values(dir_pullZ)
    modified_dir_Xincreas = modify_values(dir_Xincreas)
    modified_dir_finalYZ = modify_values(dir_finalYZ)

    print("Modified dir_pullZ:", modified_dir_pullZ)
    print("Modified dir_Xincreas:", modified_dir_Xincreas)
    print("Modified dir_finalYZ:", modified_dir_finalYZ)
        
    cnt_files = count_files(FOLDER_PATH)
    file_name = f"FLAP_{dur}_#{cnt_files}.csv"
    csv_file_path = os.path.join(FOLDER_PATH, file_name)
    
    print("pinting in", csv_file_path)
    do_FLAPS(dur, csv_file_path, selection_vector, modified_dir_Xincreas, modified_dir_pullZ, modified_dir_finalYZ, limits, GOAL, APPROACH)

def FLAP_minus(dur, GOAL, APPROACH, FOLDER_PATH):

    # FORCE SETTINGS
    selection_vector    = [1,   0,  1,  0, 1, 0]
    dir_pullZ           = [-4,  0, -15, 0, 0, 0]
    dir_Xincreas        = [-16, 0, -15, 0, 0, 0]
    dir_finalYZ         = [-4,  0, - 3,  0, 0, 0]
    force_type          = 2
    limits              = [8, 0.01, 8, 0.1, 0.1, 0.2]


    # Modify values
    modified_dir_pullZ = modify_values(dir_pullZ)
    modified_dir_Xincreas = modify_values(dir_Xincreas)
    modified_dir_finalYZ = modify_values(dir_finalYZ)

    print("Modified dir_pullZ:", modified_dir_pullZ)
    print("Modified dir_Xincreas:", modified_dir_Xincreas)
    print("Modified dir_finalYZ:", modified_dir_finalYZ)
        
    cnt_files = count_files(FOLDER_PATH)
    file_name = f"FLAP_{dur}_#{cnt_files}.csv"
    csv_file_path = os.path.join(FOLDER_PATH, file_name)
    
    print("pinting in", csv_file_path)
    do_FLAPS(dur, csv_file_path, selection_vector, modified_dir_Xincreas, modified_dir_pullZ, modified_dir_finalYZ, limits, GOAL, APPROACH)
    
def do_FLAPS(dur, csv_file_path, selection_vector, dir_Xincreas, dir_pullZ, dir_finalYZ, limits, GOAL, APPROACH):
    
    
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ["Timestamp", "Pose_X", "Pose_Y", "Pose_Z", "Pose_Rx", "Pose_Ry", "Pose_Rz", "Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z", "Q_w1", "Q_w2", "Q_w3"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write CSV header
        writer.writeheader()

        # FIRST MOVES 
        rtde_c.moveJ_IK(GOAL, speed, accel)

        #  ZEROING FTSENSOR
        rtde_c.zeroFtSensor()
        time.sleep(.1)
        gripper.move_and_wait_for_pos(230, 180, 100)
        time.sleep(.1)
        
        # Record data for a specified duration (e.g., 10 seconds)
        start_time = time.time()
        target_frequency = 500  # Hz
        i = 0
       
        print("Registro da ora: ", start_time)
        while (i<dur):
            # Record start time for this iteration
            iteration_start_time = timeit.default_timer()

            # Execute 500Hz control loop for 4 seconds, each cycle is 2ms
            t_start = rtde_c.initPeriod()
            
            task_frame = rtde_r.getActualTCPPose()  #is updated time by time
            if i<500:
                #dir_pullZ           = [0,  0, -10, 0, 0, 0]
                rtde_c.forceMode(task_frame, selection_vector, dir_pullZ, 2, limits)
            elif i<1350 and i>=500:
                rtde_c.forceMode(task_frame, selection_vector, dir_Xincreas, 2, limits)
            else:
                rtde_c.forceMode(task_frame, selection_vector, dir_finalYZ, 2, limits)
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
        
    gripper.move_and_wait_for_pos(0, 80, 100)
    time.sleep(0.4)
        
    rtde_c.moveJ(APPROACH, speed, accel)
    
    # Prompt user for input
    user_input = get_user_input()
    #user_input = 0.0
    
    # Open the existing CSV file and read its contents
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Add a new column header (assuming the new column is the 14th column)
    data[0].append('Y')

    # Add user input as the value in the first cell of the new column
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
 
    FOLDER_PATH1 = "//home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/FLAP/0_1"
    FOLDER_PATH2 = "//home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/FLAP/1_0"

    FLAPS_APP_q     =  [-1.611379, -1.204383, 1.543083, 4.309489, -1.596299, -0.062766] # 0 1 2
    #FLAPS_APP_q     =  [-1.822299, -1.323947, 1.674859, 4.292984, -1.42203, -0.197935] # 3 and F
    FLAPS_0_p       =  [-0.131814, 0.588514, -0.046268, -3.061561, 0.085233, -0.580164]
    FLAPS_1_p       =  [-0.103489, 0.588933, -0.037485, -3.079088, -0.000637, -0.301029]
    FLAPS_2_p       =  [-0.07345, 0.591552, -0.034033, -3.133595, 0.043438, 0.080362]
    FLAPS_3_p       =  [-0.043179, 0.591177, -0.035349, 3.120214, 0.019348, -0.363272]
    FLAPS_F_p       =  [-0.016647, 0.590711, -0.047151, -3.016676, 0.05602, 0.625068]


    # JOINT POSE POINTS _q
    APPROACH = FLAPS_APP_q
    GOAL1    = FLAPS_0_p
    GOAL2    = FLAPS_1_p
    TWO = 1
    
    #rtde_c.moveJ(HOME_q) 
    gripper.move_and_wait_for_pos(0, 80, 100)
    rtde_c.moveJ(APPROACH, speed, accel)


    for dur in range(1400, 1900, 100):
        if TWO !=2:
            print('GOAL1 --> dur:', dur)
            FLAP_plus(dur, GOAL1, APPROACH, FOLDER_PATH1)
            print('')
        
        if GOAL1 != GOAL2 and TWO == 1 or TWO==2:
            print('GOAL2 -->  dur:', dur)
            FLAP_minus(dur, GOAL2, APPROACH, FOLDER_PATH2)
            print('')
        dur += 100
        time.sleep(1.2)

    # Disconnect from the robot
    rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()
    

if __name__ == "__main__":
    main()