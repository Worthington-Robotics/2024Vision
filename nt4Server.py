from ntcore import NetworkTableInstance
import sys
inst: NetworkTableInstance = NetworkTableInstance.getDefault()

inst.startServer()

try:
	while True:
		sys.stdout.write(f"\rConnected: {inst.isConnected()} ")
		sys.stdout.flush()
except Exception as e:
	print(e)

inst.stopServer()