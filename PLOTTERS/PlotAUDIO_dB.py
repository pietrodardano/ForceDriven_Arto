import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def list_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"Input Device id {i} - {device_info.get('name')}")
    p.terminate()

def get_sound_level(data, rate):
    audio_data = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_data**2))
    db = 20 * np.log10(rms + 1e-10)
    return db

def record_sound(mic_index, rate=44100, chunk=512):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=chunk)

    fig, ax = plt.subplots()
    x_data, y_data = [], []
    ln, = plt.plot([], [], 'b-')

    def init():
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 100)
        return ln,

    def update(frame):
        data = stream.read(chunk, exception_on_overflow=False)
        db = get_sound_level(data, rate)
        current_time = time.time() - start_time
        x_data.append(current_time)
        y_data.append(db)
        if len(x_data) > 200:
            x_data.pop(0)
            y_data.pop(0)
        ax.set_xlim(max(0, current_time - 10), current_time)
        ln.set_data(x_data, y_data)
        return ln,

    start_time = time.time()
    ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

    plt.xlabel('Time (s)')
    plt.ylabel('dB')
    plt.title('Real-time Sound Level Monitoring')
    plt.show()

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    print("Available input devices:")
    list_input_devices()
    mic_index = int(input("Enter the device index of the microphone you want to use: "))
    record_sound(mic_index)