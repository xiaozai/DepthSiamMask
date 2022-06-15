import sys
import torch
sys.path.append('/home/yan/Data2/d3s/')


pretrained_path = 'DeT_DiMP50_Max.pth'
pretrained_dict = torch.load(pretrained_path)
''' No need the model '''
pretrained_dict = pretrained_dict['net']
new_dict = {}
for key in pretrained_dict.keys():
    # print(key)
    if key.startswith('feature_extractor_depth'):
        print(key[24:])
        new_dict[key[24:]] = pretrained_dict[key]
torch.save(new_dict, 'Depth_Resnet50.pth')
