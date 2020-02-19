import zmq

def remote_send(socket, data):
    try:
        socket.send_string(data)
        msg = socket.recv_string()
        return (msg)
    except zmq.Again as e:
        print ("Waiting for PUSH from MetaTrader 4..")

# Get zmq context
context = zmq.Context()

# Create REQ Socket
req = reqSocket = context.socket(zmq.REQ)
connect = reqSocket.connect("tcp://localhost:8080")

# Send RATES command to ZeroMQ MT4 EA
print('print')
while(True):
    print(remote_send(reqSocket, "RATES|LTCUSD"))

# bid, ask, buy_volume, sell_volume, tick_volume, real_volume, buy_volume_market, sell_volume_market