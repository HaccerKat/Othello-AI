import time
from multiprocessing import Process

def do_something():
    print("I'm going to sleep")
    time.sleep(1)
    print("I'm awake")

process_1 = Process(target=do_something)
process_2 = Process(target=do_something)

# Starts both processes
process_1.start()
process_2.start()

process_1.join()
process_2.join()