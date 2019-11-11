import os

name_model = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
name_dataset = "ExDark"

command_hub = ""
command_hub = command_hub+"python inference_general.py "
command_hub = command_hub+"--name_model "+name_model+" "
command_hub = command_hub+"--name_dataset "+name_dataset+" "
command_hub = command_hub+"--idx_start "

for idx in range(2):
	idx_start = str(idx*100)
	idx_end = str(idx*100+100)
	command_idx = command_hub+idx_start+" --idx_end "+idx_end
	print(command_idx)
	os.system(command_idx)
