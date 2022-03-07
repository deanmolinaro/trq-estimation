from tcpip import ClientTCP
import random
import time


client = ClientTCP('192.168.1.2', 50050)
print('Starting!')
num_vals = 22

start_time = time.perf_counter()
for i in range(2000):
    message = '!' + ','.join([str(round(random.random(),3)) for i in range(num_vals)] + [str(time.time())]) + ','

    time_start = time.perf_counter()
    client.to_server(message)
    # msg = client.from_server()
    msg = client.from_server_wait()
    print(msg, time.perf_counter() - time_start)
    time.sleep(0.005)
# print(time.perf_counter()-start_time)