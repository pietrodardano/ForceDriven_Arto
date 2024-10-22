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

def doButton(fZ, dur, GOAL, APPROACH, FOLDER_PATH):

    # FORCE SETTINGS
    selection_vector    = [1, 0,  0, 0, 0, 0]
    dir_force           = [fZ, 0,  0, 0, 0, 0]
    force_type          = 2
    limits              = [0.04, 0.01, 0.01, 0.01, 0.01, 0.01]
    
    cnt_files = count_files(FOLDER_PATH)
    file_name = f"Lever_{fZ}N_{dur}_#{cnt_files}.csv"
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
    

    FOLDER_PATH1 = "/home/user/thesis_ws/src/LEVER_GRIPA320/FRAME/"
    FOLDER_PATH2 = "/home/user/thesis_ws/src/LEVER_GRIPA320/FCU_APPR/"

    # JOINT POSE POINTS _q
    APPROACH = [-1.349006, -0.772585, 1.278837, 3.370017, -0.391193, -0.735367]
    GOAL1    = [-0.495083, 0.672675, -0.032743, -2.444092, -1.651991, 0.094095]
    GOAL2    = [-0.54498, 0.610219, 0.484106, -1.446434, -1.427549, 0.962667] # FCU_APPR
    TWO = 0
    
    rtde_c.moveJ(HOME_q) 
    rtde_c.moveJ(APPROACH, speed, accel)
    Zvalue = [8,9,10]

    if(1):
        for fZ in Zvalue:
            print('---- now ', fZ, 'N ----')
            dur =  800
            
            while dur < 2200 :
                print('GOAL1 --> force:', fZ, '   dur:', dur)
                result_file_path = doButton(fZ, dur, GOAL1, APPROACH, FOLDER_PATH1)
                print('')
                
                data = pd.read_csv(result_file_path)
                force_z_values = data['Force_Z'].unique()
                #rtde_c.forceModeSetGainScaling(0.1)

                # Check if all force_z values are equal
                if len(force_z_values) == 1:
                    file_name = os.path.basename(result_file_path)
                    print(f"Deleting file: {file_name}")
                    os.remove(result_file_path)
                    break

                if GOAL1 != GOAL2 and TWO == 1:
                    print('GOAL2 --> force:', fZ, '   dur:', dur)
                    result_file_path = doButton(fZ, dur, GOAL2, APPROACH, FOLDER_PATH2)
                    print('')
                    data = pd.read_csv(result_file_path)
                    force_z_values = data['Force_Z'].unique()
                
                    # Check if all force_z values are equal
                    if len(force_z_values) == 1:
                        file_name = os.path.basename(result_file_path)
                        print(f"Deleting file: {file_name}")
                        os.remove(result_file_path)
                        break

                dur += 200
                time.sleep(1)
    else:
        for i in range(1):
            doButton(8, 800, GOAL1, APPROACH, FOLDER_PATH1)
            c = input('gnegne --- ')
            time.sleep(0.8)

    # Disconnect from the robot
    rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()
    

if __name__ == "__main__":
    main()