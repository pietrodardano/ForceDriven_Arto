# Guide to integration of Pietro's Thesis into ARTO

#### For any question: dardanopietro@libero.it

### Each action --> it's own Classification model:
- Buttons
- Knobs
- Switches
- FLAP
- LDG 
- Speed Brake

## ROBOT ACTION MODIFICATION
Integrate the data registration while executing the action. In my case i saved everything in CSV file, but instead you can just create NUMPY vectors (arrays) that will last only for that phase, then are deleted (saving memory) but will still appear in the ARTO_LOG, later on.
```
    file_name = f"Butt_{fZ}N_{dur}_#{cnt_files}.csv"
    csv_file_path = os.path.join(FOLDER_PATH, file_name)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ["Timestamp", "Pose_X", "Pose_Y", "Pose_Z", "Pose_Rx", "Pose_Ry", "Pose_Rz",Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write CSV header
        writer.writeheader()

        # FIRST MOVES 
        rtde_c.moveJ_IK(GOAL, speed, accel)
        time.sleep(.1)
        
        # Record data for a specified duration (e.g., 10 seconds)
        start_time = time.time()
        target_frequency = 500  # Hz
        i = 0   # Time index

        #  ZEROING FTSENSOR
        rtde_c.zeroFtSensor()
        print("Registro da ora: ", start_time)

        while (i<dur):      # dur == duration, must be passed in

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
```
Basically it uses the rtde_r to acquire Forces, Torques, TCP Pose, JointAngles (even if not used in the classification) to be later used, some of them, for the classification task

INCLUDE THE FOLLOWING ONLY IN CASE TO CREATE/AUGMENT THE DATASET, THIS PART SERVES TO SAVE THE LABEL (Ground-Truth) USED FOR THE MANUAL LABELLING PROCESS OF EACH ACTION.
```
    # Prompt user for input
    user_input = get_user_input()
    
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
```

now the FEEDBACKS from the interaction are ready to be used for dataset augmentation or for the direct classification of the action.

# GENERAL PREPROCESSING INFO:

Every data-signal is filtered at 30Hz via a 2nd order Butterworth filter (use Scipy)

if PADDING is employed it repeats the last valid value + white noise
the added white noise has 0 mean and standard deviation equal to np.std(noisy_signal - filtered_signal)

## Transformation (Rotation) of Reference-Frame: Base --> TCP
THE FORCE AND TORQUE SIGNALS MUST BE TRANSFORMED WRT THE TCP-REFERENCE-FRAME 
since are returned from the robot respect the base reference frame !!!

```
def transform_force_torque_to_tcp(tcp_pose, force_base):
    rotation_vector = np.array(tcp_pose[3:])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    force_base_vector = np.array(force_base[:3])
    torque_base_vector = np.array(force_base[3:])
    force_tcp_vector = np.dot(rotation_matrix.T, force_base_vector)
    torque_tcp_vector = np.dot(rotation_matrix.T, torque_base_vector)
    
    return np.concatenate((force_tcp_vector, torque_tcp_vector))

# TO PUT THIS FOLLOWING CODE INSIDE THE PREPROCESSING FUNCTION
        # Convert data to NumPy arrays for faster processing
        tcp_pose = data[['Pose_X', 'Pose_Y', 'Pose_Z', 'Pose_Rx', 'Pose_Ry', 'Pose_Rz']].to_numpy()
        forces = data[['Force_X', 'Force_Y', 'Force_Z']].to_numpy()
        torques = data[['Torque_X', 'Torque_Y', 'Torque_Z']].to_numpy()
        y = data['Y'].values[0]

        # Transform forces and torques to the TCP frame before preprocessing
        transformed_signals = []
        for i in range(len(data)):
            transformed_signal = transform_force_torque_to_tcp(tcp_pose[i], np.concatenate((forces[i], torques[i])))
            transformed_signal[2] = -transformed_signal[2]  # Invert the sign of Force_Z_TCP after the transformation
            transformed_signals.append(transformed_signal)
        
        transformed_signals = np.array(transformed_signals)

```

## BUTTON PREPROCESSING

Buttons actions data are NOT normalized, trimmed to 800 of duration even if the action was longer
the trimming phase occours after selecting and isolating the MAIN TRANSIENT of the action, it's based on the energy of the signal + 3 thresholds.

### IF THE GAIN OF THE FORCEMODE ISN'T MODIFIED 
the "energy-based transient detection" is required to isolate the most information-rich part of the signal, thus trimming it. Otherwise the ML model will have input a signal that presents oscillations and those can mislead the classification result to an incorrect one.

### TRANSIENT DETECTION AND ISOLATION
ONLY for the BUTTON action, all the other ones does not have it.

```
def sliding_sum_window(signal, squared_signal_start, window_size=ENERGY_WS*FS_UR5E, accept_factor=0.25):
    len_sign = len(signal)
    max_sum = 0
    window_size = int(window_size)
    
    # find max (energy/window_sum)
    for i in range(len_sign-window_size):
        window = signal[i:i+window_size]
        summed = np.sum(window)
        if summed>max_sum: 
            max_sum=summed
            #max_sum_index=i
    min_sum_acepted = max_sum*accept_factor # % of the max summed values
    
    #print(squared_signal_start)
    for index in squared_signal_start:
        if np.sum(signal[index: min(index+window_size, len(signal))])>min_sum_acepted:
            return index
    
    return 0

# ALGORITHM TO SELECT THE INDEX RETURNED BY THE THRESHOLD OVER THE ENERGY OF THE SIGNAL
# TO DETECT AND ISOLATE THE CORRECT PART OF IT.
# THRESHOLDS: 22-28-38 %
def select_index(a, b, c, signal):
    vals = [a,b,c]
    vals = sorted(vals, reverse=True)
    a = vals[0]
    b = vals[1]
    c = vals[2]
    # SO A>B>C
    if np.abs(a - b) < 60 and np.abs(b - c) < 60:
        res = max(b - DISTANCE_SAMPLES, 0)
        if res > len(signal) - WS_B:    res = len(signal) - WS_B
        return int(res)
    else:
        res = max(c - DISTANCE_SAMPLES, 0)
        if res > len(signal) - WS_B:    res = len(signal) - WS_B
        return int(res)
```

## ALL OTHER ACTIONS

Durations:
```
LEVER   -> 2000
KNOBS   -> 2000
LDG     -> 3000
FLAP    -> 1800
SPEED   -> 1800
```
These values are the "Target Length" thus even if the real action is shorter (longer) is then padded (trimmed) to reach the target length to have consistent sizes to be given input to the ML models either for the direct classification either for a new training.

The preprocessing consists in:
1) Transformation (of reference frame), NOT THE POSES!
2) Filtering (30 Hz)
3) Padding - Truncation to desider length
4) Normalization (Standard Scaler)

you can find all the sequence and functions in the dedicated PREPROCESSING script.

# DATASET MANAGEMENT

You will find 2 Stages: 
1) DATASET CREATION AND LOADING FOR THE TRAINING
2) DATA LOADING FOR THE CLASSIFICATION (Implementable in ARTO Poduct)

## Case (1): Dataset Creation for Training

In sinthesys is: **CSV** (registered from the robot) --> **Numpy** [.npy.npz] (after preprocessing) --> **Numpy** (selected and splitted Test-Valid-Train) then **loaded into the GPU** for the training.

Again, here i used the CSV for the general dataset, directly created from the robot acquisitions but it can be used any other type of data storing.

Then all the data are preprocessed and saved as numpy arrays [.npy] and compressed [.npz] and stored in the same main folder in which the ML models are contained.

Then to load all the data, the signals are extracted and loaded (in the jupyter-python script of the ML models)
Here the data are then SELECTED (e.g. using only Fx, Fz, Ty) and SPLITTED (60%-20%-20%) for Train-Val-Test
```
def load_data(data_folder):
    X_data = []
    y_data = []
    
    # Traverse the data folder
    for file in os.listdir(data_folder):
        if file.endswith(".npz"):
            file_path = os.path.join(data_folder, file)
            data = np.load(file_path)
            X_data.append(data['X'])
            y_data.append(data['y'])
    
    # Stack the data into arrays
    X_data = np.vstack(X_data)
    y_data = np.hstack(y_data)
    
    return X_data, y_data

TEST_SIZE = 0.4
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=31)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=31)

``` 

## Case (2): Data loading for direct classification

Robot records data --> Convertion to **.npy**--> **Preprocessing** --> **Selection** --> Input to ML models

EXAMPLE TO SAVE THEM DIRECTLY TO NPY:
```
timestamps = []
pose_x = []
pose_y = []
pose_z = []
pose_rx = []
pose_ry = []
pose_rz = []
force_x = []
force_y = []
force_z = []
torque_x = []
torque_y = []
torque_z = []

# Record data for a specified duration (e.g., 10 seconds)
start_time = time.time()
target_frequency = 500  # Hz
i = 0   # Time index

# ZEROING FTSENSOR
rtde_c.zeroFtSensor()
print("Recording started at: ", start_time)

while i < dur:  # dur == duration, must be passed in

    # Record start time for this iteration
    iteration_start_time = timeit.default_timer()

    # Execute 500Hz control loop for 4 seconds, each cycle is 2ms
    t_start = rtde_c.initPeriod()

    task_frame = rtde_r.getActualTCPPose()  # updated in real-time
    rtde_c.forceMode(task_frame, selection_vector, dir_force, force_type, limits)
    rtde_c.waitPeriod(t_start)

    # Read force data
    force_data = rtde_r.getActualTCPForce()
    pose_data = rtde_r.getActualTCPPose()

    timestamp = time.time() - start_time

    # Store the data in lists
    timestamps.append(timestamp)
    pose_x.append(pose_data[0])
    pose_y.append(pose_data[1])
    pose_z.append(pose_data[2])
    pose_rx.append(pose_data[3])
    pose_ry.append(pose_data[4])
    pose_rz.append(pose_data[5])
    force_x.append(force_data[0])
    force_y.append(force_data[1])
    force_z.append(force_data[2])
    torque_x.append(force_data[3])
    torque_y.append(force_data[4])
    torque_z.append(force_data[5])

    # Calculate time taken for this iteration
    iteration_time = timeit.default_timer() - iteration_start_time

    # Adjust sleep duration for target frequency
    sleep_duration = 1.0 / target_frequency - iteration_time
    if sleep_duration > 0:
        time.sleep(sleep_duration)
    
    i += 1

rtde_c.forceModeStop()

# Convert lists to NumPy arrays
data = np.array([timestamps, pose_x, pose_y, pose_z, pose_rx, pose_ry, pose_rz,
                 force_x, force_y, force_z, torque_x, torque_y, torque_z])

# Save the data to a NumPy file
np.save('force_pose_data.npy', data)

```

now you have **.NPY** data

At this point just preprocess it as earlier indicated, depending if is a button or not.

then you will have a ROS node (or something else) that has the ML model loaded in it, if you pass to it the selected data in input, it will then classify it and based on the employed model you have to select the respective data.

# LABELS and Cases
The output values (Integers) have specific meanings, one for each case as below reported:

#### Button
- 0: Failed the pressing action, of pressed the frame of the strumentation
- 1: Successfully pressed ( !! 1 or multiple times !! )

#### KNOB
- 0: Fail, not turned enough to reach the following state
- 1: Turned correctly by 1 state and engaged
- 2: Turned correctly by 2 states and engaged
- 3: Turned correctly by 3 states and engaged
- 4: Turned correctly by 1 but OVERTURNED so now its between 1° and 2° state, not engaged
- 5: Turned correctly by 1 but OVERTURNED so now its between 2° and 3° state, not engaged

#### SWITCH (called LEVER in the scripts/models)
- 0: Fail, not moved
- 1: Correct 1° state engaged
- 2: Correct 2° state engaged 

#### FLAP
- 0: Fail, not pulled enough, not unlocked -- Problem at the begin of the action
- 1: Fail, not moved enough laterally, the lever return to the previous state or block in the between
- 2: Correct, moved correctly to the next state and set correctly

#### LDG
- 0: Fail, not pulled enough, not unlocked -- Problem at the begin of the action
- 1: Fail, not moved enough laterally
- 2: Correct BUT released badly, abruptly 
- 3: Correct and gently released (like a human)

#### SPEED BRAKE
- 0: Fail, not unlocked, moved in wrong direction or not clamped the object -- Problem at the begin of the action
- 1: Fail, not moved enough laterally to reach 1° or 2° state
- 2: Fail, moved too much to engage the middle state, bypassing it
- 3: Correct, moved correctly to the 1° or 2° state and set correctly

# eXplanable AI: Grad-CAM

1D GradCAM is implemented for multibranch models.
it plots an heatmap, over the plot of the 1D signals given input to the model, to highlight the part of the input signals used more by the ML model to perform the classification.
For an experienced user it allows to check if the model is focusing on the right features of the signals to classify it well and coherently.

e.g: the model is focusing towards the end of a signal where the part of the signals are referred to the end of the action, thus the release or the final motions; if the models classify it as "correct" or "moved not enough" it could be OK, but if instead it has been classified as "it didn't unlock the lever at the beginning" but, but still focuses towards the end, there is something bad that happened and lead the model to mislead the classification.

The plot+heatmaps (by GradCam) should be integrated in the ARTO_LOG too.
the Arto_user could benefit of these infos to chec if the model classified it correctly or not.
It's a feature to make the whole process a bit more robust. 

```
def compute_grad_cam(model: Model, inputs: List[np.ndarray], last_layers_names: str, class_idx: int, epsilon: float = 1e-9) -> np.ndarray:
    model = Model(inputs=model.inputs, outputs=[model.get_layer(last_layers_names).output, model.output])
    with tf.GradientTape() as tape:
        output, predictions = model(inputs)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, output) #  list or nested structure of Tensors (or IndexedSlices, or None, or CompositeTensor), one for each element in sources.
    
    #reduced_grads == pooled_grads 
    reduced_grads = tf.reduce_mean(grads, axis=(0, 1))
    output = output[0] #otherwise it will remain a tensor, not a 1D 
    
    #make them workable by numpy 
    reduced_grads = reduced_grads.numpy()
    output = output.numpy()

    for i in range(reduced_grads.shape[-1]):
        output[:, i] *= reduced_grads[i]        #obtaining an heatmap
    
    heatmap = np.mean(output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0: heatmap /= np.max(heatmap) #to avoid zero division
    return heatmap
```
#### THE FOLLOWING CODE IS TO PLOT GRAD-CAM FOR EACH INPUT/BRANCH OF THE MULTI-BRANCH MODELS

```
def plot_grad_cam(model: Model, 
                  X_tests: List[Union[np.ndarray, np.ndarray]], 
                  sample_idx: int, 
                  y_test: np.ndarray, 
                  conv_layers: List[str],
                  labels: Optional[List[str]] = None) -> None:

    # Prepare inputs
    try:
        inputs = [np.expand_dims(X_test[sample_idx], axis=0) for X_test in X_tests]
    except IndexError as e:
        print(f"Error: {e}. Check if sample_idx {sample_idx} is within the range of X_tests.")
        return
    
    # Predict class index and probabilities
    y_pred_prob = model.predict(inputs)
    class_idx = np.argmax(y_pred_prob, axis=1)[0]
    y_pred_label = class_idx if y_pred_prob.shape[1] > 1 else (y_pred_prob[0][0] > 0.5).astype(int)

    # Compute Grad-CAM heatmaps
    heatmaps = [compute_grad_cam(model, inputs, conv_layer, class_idx) for conv_layer in conv_layers]

    # Plot Grad-CAM heatmaps
    plt.figure(figsize=(12, 3 * len(X_tests)))

    if labels is None:
        labels = [f'Branch {i+1}' for i in range(len(X_tests))]

    for i, (X_test, heatmap) in enumerate(zip(X_tests, heatmaps)):
        plt.subplot(len(X_tests), 1, i + 1)
        #plt.title(f'Grad-CAM for {labels[i]} signal')
        plt.title(f'Grad-CAM for {labels[i]}')
        
        if X_test.ndim == 2:  # Single channel input
            plt.plot(X_test[sample_idx])
            plt.imshow(heatmap[np.newaxis, :], aspect="auto", cmap='summer', alpha=0.6,
                       extent=(0, X_test.shape[1], np.min(X_test[sample_idx]), np.max(X_test[sample_idx])))
            plt.ylim(np.min(X_test[sample_idx]) - 0.0 * np.abs(np.min(X_test[sample_idx])), 
                     np.max(X_test[sample_idx]) + 0.0 * np.abs(np.max(X_test[sample_idx])))
            #plt.ylabel('[m]')
            if i == len(X_tests) - 1:  # Only show x-label for the last subplot
                plt.xlabel('Samples [2ms]')
        elif X_test.ndim == 3:  # Multi-channel input
            for channel in range(X_test.shape[-1]):
                plt.plot(X_test[sample_idx, :, channel], label=f'Signal {channel+1}')
            plt.imshow(heatmap[np.newaxis, :], aspect="auto", cmap='summer', alpha=0.6,
                       extent=(0, X_test.shape[1], np.min(X_test[sample_idx]), np.max(X_test[sample_idx])))
            plt.ylim(np.min(X_test[sample_idx]) - 0.0 * np.abs(np.min(X_test[sample_idx])), 
                     np.max(X_test[sample_idx]) + 0.0 * np.abs(np.max(X_test[sample_idx])))
            # if i == len(X_tests) - 1: 
            #     plt.ylabel('[m]')
            # else:
            #     plt.ylabel('[N]')
            if i == len(X_tests) - 1:  # Only show x-label for the last subplot
                plt.xlabel('Samples [2ms]')
            plt.legend()
        
        plt.colorbar()
    
    plt.suptitle(f"Test data number: {sample_idx} --> Yreal: {y_test[sample_idx]}, Ypred: {y_pred_label}")
    plt.tight_layout()
    plt.show()
```
