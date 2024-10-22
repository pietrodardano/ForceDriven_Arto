import os
import csv
import time
import timeit
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
    user_input = input("Was it successfull? Y or N: ").upper()
    
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

def doButton(dur, GOAL, APPROACH, FOLDER_PATH):

    # FORCE SETTINGS
    selection_vector    = [1,  1,  1,  0, 1, 0]
    dir_pullZ           = [4,  -1, -29, 0, 0, 0]
    dir_Xincreas        = [22, 0, -28, 0, 0, 0]
    dir_finalYZ         = [0,  0, -4,  0, 0, 0]
    force_type          = 2
    limits              = [8, 0.01, 8, 0.1, 0.1, 0.2]
    
    cnt_files = count_files(FOLDER_PATH)
    file_name = f"FLAP_{dur}_#{cnt_files}.csv"
    csv_file_path = os.path.join(FOLDER_PATH, file_name)
    
    print("pinting in", csv_file_path)
    # Open CSV file for writing
    
    
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ["Timestamp", "Pose_X", "Pose_Y", "Pose_Z", "Pose_Rx", "Pose_Ry", "Pose_Rz", "Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write CSV header
        writer.writeheader()

        # FIRST MOVES 
        rtde_c.moveJ_IK(GOAL, speed, accel)

        #  ZEROING FTSENSOR
        rtde_c.zeroFtSensor()
        time.sleep(.1)
        gripper.move_and_wait_for_pos(230, 180, 150)
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
                rtde_c.forceMode(task_frame, selection_vector, dir_pullZ, force_type, limits)
            elif i<1200 and i>=500:
                rtde_c.forceMode(task_frame, selection_vector, dir_Xincreas, force_type, limits)
            else:
                rtde_c.forceMode(task_frame, selection_vector, dir_finalYZ, force_type, limits)
            rtde_c.waitPeriod(t_start)

            # Read force data
            force_data = rtde_r.getActualTCPForce()
            pose_data  = rtde_r.getActualTCPPose()

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
                "Torque_Z": force_data[5]
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
 
    FOLDER_PATH1 = "/home/user/arto_ws/src/RobotData_GRIPA320/MCDU_FPLN/"
    FOLDER_PATH2 = "/home/user/arto_ws/src/RobotData_GRIPA320/MCDU_INIT/"

    # JOINT POSE POINTS _q
    APPROACH = MCDU_APP_q
    GOAL1    = MCDU_FPLN_p
    GOAL2    = MCDU_INIT_p
    TWO = 0
    
    rtde_c.moveJ(HOME_q) 
    gripper.move_and_wait_for_pos(0, 80, 100)
    rtde_c.moveJ(APPROACH, speed, accel)

    for i in range(5):
        print('---- now ', i, 'th ----')
        dur =  3000
        
        while dur < 3400 :
            print('GOAL1 --> dur:', dur)
            result_file_path = doButton(dur, GOAL1, APPROACH, FOLDER_PATH1)
            print('')
            
            # data = pd.read_csv(result_file_path)
            # force_z_values = data['Force_Z'].unique()
            # #rtde_c.forceModeSetGainScaling(0.1)
            # # Check if all force_z values are equal
            # if len(force_z_values) == 1:
            #     file_name = os.path.basename(result_file_path)
            #     print(f"Deleting file: {file_name}")
            #     os.remove(result_file_path)
            #     break
            if GOAL1 != GOAL2 and TWO == 1:
                print('GOAL2 -->  dur:', dur)
                result_file_path = doButton(dur, GOAL2, APPROACH, FOLDER_PATH2)
                print('')
                # data = pd.read_csv(result_file_path)
                # force_z_values = data['Force_Z'].unique()
            
                # # Check if all force_z values are equal
                # if len(force_z_values) == 1:
                #     file_name = os.path.basename(result_file_path)
                #     print(f"Deleting file: {file_name}")
                #     os.remove(result_file_path)
                #     break
            dur += 100
            time.sleep(1.2)

    # Disconnect from the robot
    rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()
    

if __name__ == "__main__":
    main()