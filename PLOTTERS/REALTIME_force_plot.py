import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rtde_receive

robot_ip = "192.168.1.102"
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

# Initialize variables
data = {"Timestamp": [], "Torque_Z": [], "Force_X": [], "Torque_X": [], "Torque_Y": [], "Force_Y": []}

# Function to update plot
def update(frame):
    # Get force/torque data
    force_data = rtde_r.getActualTCPForce()
    timestamp = frame
    
    # Extract data
    torque_z = force_data[5]
    force_x = force_data[0]
    torque_x = force_data[3]
    torque_y = force_data[4]
    force_y = force_data[1]
    
    # Append data to dictionary
    data["Timestamp"].append(timestamp)
    data["Torque_Z"].append(torque_z)
    data["Force_X"].append(force_x)
    data["Torque_X"].append(torque_x)
    data["Torque_Y"].append(torque_y)
    data["Force_Y"].append(force_y)
    
    # Update subplot 1: Torque_Z
    ax1.clear()
    ax1.plot(data["Timestamp"], data["Torque_Z"], label="Torque_Z")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Torque_Z")
    ax1.set_title("Real-time Plot of Torque_Z")
    ax1.grid(True)
    ax1.legend()
    
    # Update subplot 2: Force_X
    ax2.clear()
    ax2.plot(data["Timestamp"], data["Force_X"], label="Force_X")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Force_X")
    ax2.set_title("Real-time Plot of Force_X")
    ax2.grid(True)
    ax2.legend()
    
    # Update subplot 3: Torque_X
    ax3.clear()
    ax3.plot(data["Timestamp"], data["Torque_X"], label="Torque_X")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Torque_X")
    ax3.set_title("Real-time Plot of Torque_X")
    ax3.grid(True)
    ax3.legend()
    
    # Update subplot 4: Torque_Y
    ax4.clear()
    ax4.plot(data["Timestamp"], data["Torque_Y"], label="Torque_Y")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Torque_Y")
    ax4.set_title("Real-time Plot of Torque_Y")
    ax4.grid(True)
    ax4.legend()
    
    # Update subplot 5: Force_Y
    ax5.clear()
    ax5.plot(data["Timestamp"], data["Force_Y"], label="Force_Y")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Force_Y")
    ax5.set_title("Real-time Plot of Force_Y")
    ax5.grid(True)
    ax5.legend()

# Create figure and axis
fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(3, 2, figsize=(12, 12))

# Animate plot
ani = FuncAnimation(fig, update, frames=np.linspace(0, 4, 400), interval=10)  # Update every 10ms

# Show plot
plt.tight_layout()
plt.show()
