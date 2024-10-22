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
    while True:
        try:
            user_input = int(input("HAS IT DONE A CLICK? Enter a number between 0 and 3: "))
            if 0 <= user_input <= 2:
                return user_input
            else:
                print("Invalid input. Please enter a number between 0 and 3.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

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

def do_LeverSx(fZ, dur, GOAL, APPROACH, FOLDER_PATH):
    if fZ < 0:
        doLever(fZ, dur, GOAL, APPROACH, FOLDER_PATH)
    elif fZ>0:
        doLever(-fZ, dur, GOAL, APPROACH, FOLDER_PATH)


def do_LeverDx(fZ, dur, GOAL, APPROACH, FOLDER_PATH):
    if fZ > 0:
        doLever(fZ, dur, GOAL, APPROACH, FOLDER_PATH)
    elif fZ<0:
        doLever(-fZ, dur, GOAL, APPROACH, FOLDER_PATH)

def doLever(fZ, dur, GOAL, APPROACH, FOLDER_PATH):

    # FORCE SETTINGS
    selection_vector    = [1, 0,  0, 0, 0, 0]
    dir_force           = [fZ, 0,  0, 0, 0, 0]
    force_type          = 2
    limits              = [4, 0.01, 0.01, 0.01, 0.01, 0.01]
    
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

        #gripper.move_and_wait_for_pos(180, 90, 0)

        # FIRST MOVES 
        rtde_c.moveJ_IK(GOAL, speed, accel)

        # ZEROING FTSENSOR
        rtde_c.zeroFtSensor()
        #gripper.move_and_wait_for_pos(180, 90, 0)
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
    #user_input = 0
    
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
    

    FOLDER_PATH1 = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/LEVER/ANN_LT/"
    FOLDER_PATH2 = "/home/user/thesis_ws/src/ROBOT_ACTIONS_DATA/LEVER/ANN_LT/"

    # GRIPPER OPEN
    APPROACH = [-1.60405, -1.065008, 1.205881, 4.399705, 1.303686, 0.102238] # mid air for ANN LT
    APPROACH = [-1.752691, -1.219639, 1.574714, 4.240185, 1.300944, 0.212839] # MID FOR DOME LT
    #APPROACH = [-1.62091, -1.372611, 1.699409, 4.344549, 1.361266, 0.065532] # mid air for SEAT BELT
    #GOAL1    = [-0.150800, 0.609342, 0.825756, -0.456731, -0.195753, 3.034695] # ANN_LT DIM
    #GOAL1    = [-0.188948, 0.542335, 0.818162, 0.466725, 0.152713, -3.099195]  # SEAT BELT OFF
    #GOAL2    = [-0.183459, 0.541398, 0.820523, -0.49318, -0.15682, 3.081804]   # SEAT BELT ON
    #GOAL2    = [-0.137007, 0.608427, 0.82868, -0.469544, -0.195258, 3.03156] # ANN_LT TEST
    #GOAL2   =  [-0.194419, 0.571369, 0.817346, 0.464823, 0.155522, -3.083588]  # SMOKE FOR 0 OF SEABELTS (DX)
    
    # GRIPPER FULL CLOSED
    #GOAL1 = [-0.165663, 0.64366, 0.823691, -0.435406, -0.150337, 3.07506] # ANN LT DIM
    #GOAL2 = [-0.124046, 0.641499, 0.836294, -0.38892, -0.152504, 3.102766] # ANN LT TEST
    GOAL1 = [-0.163189, 0.609319, 0.819000, -0.476146, -0.083882, 3.051466] # DOME OFF
    GOAL2 = [-0.121585, 0.607217, 0.833087, 0.476552, 0.083508, -3.053272] # DOME BRT
    #GOAL1 = [-0.205902, 0.544565, 0.812015, 0.439276, 0.121467, -3.047118] # SEAT OFF 
    #GOAL2 = [-0.167725, 0.545142, 0.824425, -0.482163, -0.133727, 3.078142] # SEAT ON 
    
    TWO = 1
    
    #rtde_c.moveJ(HOME_q) 
    rtde_c.moveJ(APPROACH, speed, accel)
    Zvalue = [8]

    if(1):
        for fZ in Zvalue:
            print('---- now ', fZ, 'N ----')
            dur =  1800
            
            while dur < 2100 :
                print('GOAL1 --> force:', fZ, '   dur:', dur)
                result_file_path = do_LeverSx(fZ, dur, GOAL1, APPROACH, FOLDER_PATH1)
                print('')
                
                # data = pd.read_csv(result_file_path)
                # force_z_values = data['Force_X'].unique()
                # #rtde_c.forceModeSetGainScaling(0.1)

                # # Check if all force_z values are equal
                # if len(force_z_values) == 1:
                #     file_name = os.path.basename(result_file_path)
                #     print(f"Deleting file: {file_name}")
                #     os.remove(result_file_path)
                #     break

                if GOAL1 != GOAL2 and TWO == 1:
                    print('GOAL2 --> force:', fZ, '   dur:', dur)
                    result_file_path = do_LeverDx(fZ, dur, GOAL2, APPROACH, FOLDER_PATH2)
                    print('')
                    # data = pd.read_csv(result_file_path)
                    # force_z_values = data['Force_X'].unique()
                
                    # # Check if all force_z values are equal
                    # if len(force_z_values) == 1:
                    #     file_name = os.path.basename(result_file_path)
                    #     print(f"Deleting file: {file_name}")
                    #     os.remove(result_file_path)
                    #     break

                dur += 200
                time.sleep(1)
    else:
        for i in range(1):
            doLever(8, 800, GOAL1, APPROACH, FOLDER_PATH1)
            c = input('gnegne --- ')
            time.sleep(0.8)

    # Disconnect from the robot
    rtde_c.stopScript()
    rtde_c.disconnect()
    rtde_r.disconnect()
    

if __name__ == "__main__":
    main()