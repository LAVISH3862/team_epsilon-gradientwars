"""Keep Windows from sleeping while queue.sh runs; exits (and lets the PC sleep again) once logs/queue.log says the queue
finished. Leave the laptop plugged in.  python keep_awake.py"""
import ctypes, os, time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print('keeping the PC awake until logs/queue.log says "queue finished"', flush=True)
while not (os.path.exists('logs/queue.log') and 'queue finished' in open('logs/queue.log').read()):
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
