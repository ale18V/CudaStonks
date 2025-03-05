.PHONY: gpu pytorch cpu

pull:
	python3 scripts/getData.py

gpu:	
	env PYTHONPATH="$$PYTHONPATH:src/" python3 -m CudaStonks handcrafted --gpu

cpu:	
	env PYTHONPATH="$$PYTHONPATH:src/" python3 -m CudaStonks handcrafted --cpu

pytorch:
	env PYTHONPATH="$$PYTHONPATH:src/" python3 -m CudaStonks pytorch
