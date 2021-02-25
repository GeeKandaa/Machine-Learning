from modules import json_comm
from mpi4py import MPI
import numpy
import sys, os
sys.path.append(os.getcwd())

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
status = MPI.Status()
print(status)
print("rank=",rank)
print("size=",size)

if rank==0:
	data_store=[]
	# from modules import json_comm
	# json_comm.mpi_get_param()
	for id in range(1, size):
		data_store.append(["Experiment "+str(id), comm.recv(source=id,status=status,tag=id)])
	for i in range(0,len(data_store)-1):
		print(data_store[i])
	print(data_store)
	json_comm.store_mp_data(data_store)

if rank!=0:
	import Auto_Parameterise_Image_CNN