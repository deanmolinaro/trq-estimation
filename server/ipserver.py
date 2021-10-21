import socket


class ServerTCP(object):
	def __init__(self, server_ip, recv_port):
		self.SERVER_IP = server_ip
		self.RECV_PORT = recv_port
		self.recv_conn = 0.

	def close(self):
		self.recv_conn.close()

	def from_client(self):
		return self.recv_conn.recv(8192)

	def to_client(self, msg):
		self.recv_conn.sendall(msg.encode())
		return

	def start_server(self):
		recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		recv_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		recv_socket.bind((self.SERVER_IP, self.RECV_PORT))
		recv_socket.listen(1)
		print('\nWaiting for client to connect.')
		self.recv_conn, recv_addr = recv_socket.accept()
		recv_socket.close()
		print('Client connected!')
		return

if __name__=="__main__":
	print('Initializing server.')
	server = ServerTCP('', 8080)
	server.start_server()

	while True:
		msg = server.from_client()
		if any(msg):
			msg = msg.decode()
			print(f'Received: {msg}')
			break

	print(f'Sending: {msg}')
	server.to_client(msg)