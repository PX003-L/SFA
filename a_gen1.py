
#  一个攻击算法  攻击4个代理模型 IC3 resnet50 den121 vit 跟一个 集成模型 (resnet50 den121 vit)

import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--attack', default='ni_fgsm', type=str, help='fgsm, ifgsm, mifgsm, vmi_fgsm, ssi,ssmi, si_fgsm,smi_fgsm,admix,mmi_fgsm')

# 攻击的模型 densenet121、inception_v3、resnet50、swin-t、vit -- "Ensemble_Model"---- tf2torch_ens3_adv_inc_v3 、tf2torch_ens4_adv_inc_v3 、tf2torch_ens_adv_inc_res_v2

opt = parser.parse_args()



# os.system('python ge_adv.py --attack dim --name resnet50')
# os.system('python ge_adv.py --attack tim --name resnet50')
# os.system('python ge_adv.py --attack sim --name resnet50')
# os.system('python ge_adv.py --attack admix --name resnet50')
# os.system('python ge_adv.py --attack SSA --name resnet50')
# os.system('python ge_adv.py --attack dimsfa --name resnet50')
# os.system('python ge_adv.py --attack timsfa --name resnet50')
# os.system('python ge_adv.py --attack admixsfa --name resnet50')
# os.system('python ge_adv.py --attack ssasfa --name resnet50')
# os.system('python ge_adv.py --attack simsfa --name resnet50')
#
# os.system('python ge_adv.py --attack mifgsm --name vit')
# os.system('python ge_adv.py --attack mifgsm --name densenet121')
# os.system('python ge_adv.py --attack mifgsm --name resnet50')
# os.system('python ge_adv.py --attack mifgsm --name vgg19')
#
# os.system('python ge_adv.py --attack nifgsm --name vit')
# os.system('python ge_adv.py --attack nifgsm --name densenet121')
# os.system('python ge_adv.py --attack nifgsm --name resnet50')
# os.system('python ge_adv.py --attack nifgsm --name vgg19')
#
# os.system('python ge_adv.py --attack vmifgsm --name densenet121')
# os.system('python ge_adv.py --attack vmifgsm --name resnet50')
# os.system('python ge_adv.py --attack vmifgsm --name vgg19')
# os.system('python ge_adv.py --attack vmifgsm --name vit')
#
# os.system('python ge_adv.py --attack gra --name densenet121')
# os.system('python ge_adv.py --attack gra --name resnet50')
# os.system('python ge_adv.py --attack gra --name vgg19')
# os.system('python ge_adv.py --attack gra --name vit')
#
# os.system('python ge_adv.py --attack pgn --name densenet121')
# os.system('python ge_adv.py --attack pgn --name resnet50')
# os.system('python ge_adv.py --attack pgn --name vgg19')
# os.system('python ge_adv.py --attack pgn --name vit')
#
# os.system('python ge_adv.py --attack gaa --name densenet121')
# os.system('python ge_adv.py --attack gaa --name resnet50')
# os.system('python ge_adv.py --attack gaa --name vgg19')
# os.system('python ge_adv.py --attack gaa --name vit')
#
# os.system('python ge_adv.py --attack ggs --name densenet121')
# os.system('python ge_adv.py --attack ggs --name resnet50')
# os.system('python ge_adv.py --attack ggs --name vgg19')
# os.system('python ge_adv.py --attack ggs --name vit')
#
# os.system('python ge_adv.py --attack respa --name densenet121')
# os.system('python ge_adv.py --attack respa --name resnet50')
# os.system('python ge_adv.py --attack respa --name vgg19')
# os.system('python ge_adv.py --attack respa --name vit')
#
os.system('python ge_adv.py --attack pfa --name densenet121')
os.system('python ge_adv.py --attack pfa --name resnet50')
os.system('python ge_adv.py --attack pfa --name vgg19')
os.system('python ge_adv.py --attack pfa --name vit')

#
# os.system('python ge_adv.py --attack sfa --name densenet121')
# os.system('python ge_adv.py --attack sfa --name resnet50')
# os.system('python ge_adv.py --attack sfa --name vgg19')
# os.system('python ge_adv.py --attack sfa --name vit')
# #
#
# os.system('python ge_adv.py --attack mifgsm --name Ensemble_Model')
# os.system('python ge_adv.py --attack nifgsm --name Ensemble_Model')
# os.system('python ge_adv.py --attack vmifgsm --name Ensemble_Model')
# os.system('python ge_adv.py --attack gra --name Ensemble_Model')
# os.system('python ge_adv.py --attack pgn --name Ensemble_Model')
# os.system('python ge_adv.py --attack gaa --name Ensemble_Model')
# os.system('python ge_adv.py --attack ggs --name Ensemble_Model')
# os.system('python ge_adv.py --attack respa --name Ensemble_Model')
os.system('python ge_adv.py --attack pfa --name Ensemble_Model')
os.system('python ge_adv.py --attack sfa --name Ensemble_Model')


# os.system('python ge_adv.py --attack ggs1 --name densenet121')
# os.system('python ge_adv.py --attack ggs1 --name resnet50')
# os.system('python ge_adv.py --attack ggs1 --name vgg19')
# os.system('python ge_adv.py --attack ggs1 --name vit')
# os.system('python ge_adv.py --attack ggs1 --name Ensemble_Model')