
#  一个攻击算法  攻击4个代理模型 IC3 resnet50 den121 vit 跟一个 集成模型 (resnet50 den121 vit)

import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--attack', default='ni_fgsm', type=str, help='fgsm, ifgsm, mifgsm, vmi_fgsm, ssi,ssmi, si_fgsm,smi_fgsm,admix,mmi_fgsm')

# 攻击的模型 densenet121、inception_v3、resnet50、swin-t、vit -- "Ensemble_Model"---- tf2torch_ens3_adv_inc_v3 、tf2torch_ens4_adv_inc_v3 、tf2torch_ens_adv_inc_res_v2

opt = parser.parse_args()


os.system('python ge_adv.py --attack SFA1 --name resnet50')
os.system('python ge_adv.py --attack SFA2 --name resnet50')
os.system('python ge_adv.py --attack SFA3 --name resnet50')
os.system('python ge_adv.py --attack SFA4 --name resnet50')
os.system('python ge_adv.py --attack SFA5 --name resnet50')
os.system('python ge_adv.py --attack SFA6 --name resnet50')
os.system('python ge_adv.py --attack SFA7 --name resnet50')
os.system('python ge_adv.py --attack SFA8 --name resnet50')
os.system('python ge_adv.py --attack SFA9 --name resnet50')

os.system('python ge_adv.py --attack SFA01 --name resnet50')
os.system('python ge_adv.py --attack SFA02 --name resnet50')
os.system('python ge_adv.py --attack SFA03 --name resnet50')
os.system('python ge_adv.py --attack SFA04 --name resnet50')
os.system('python ge_adv.py --attack SFA05 --name resnet50')
os.system('python ge_adv.py --attack SFA06 --name resnet50')
os.system('python ge_adv.py --attack SFA07 --name resnet50')
os.system('python ge_adv.py --attack SFA08 --name resnet50')
os.system('python ge_adv.py --attack SFA09 --name resnet50')
os.system('python ge_adv.py --attack SFA010 --name resnet50')
os.system('python ge_adv.py --attack SFA011 --name resnet50')

os.system('python ge_adv.py --attack SFA001 --name resnet50')
os.system('python ge_adv.py --attack SFA002 --name resnet50')
os.system('python ge_adv.py --attack SFA003 --name resnet50')
os.system('python ge_adv.py --attack SFA004 --name resnet50')
os.system('python ge_adv.py --attack SFA005 --name resnet50')
os.system('python ge_adv.py --attack SFA006 --name resnet50')
os.system('python ge_adv.py --attack SFA007 --name resnet50')
os.system('python ge_adv.py --attack SFA008 --name resnet50')
os.system('python ge_adv.py --attack SFA009 --name resnet50')
os.system('python ge_adv.py --attack SFA0010 --name resnet50')
os.system('python ge_adv.py --attack SFA0011 --name resnet50')