import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2
# 去雾
def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)
    # cv2.imshow('20190708_Dark',Dark_Channel)    # 查看暗通道
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A


def deHaze(m, r=32, eps=0.001, w=0.4, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)



	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	# data_lowlight = data_lowlight.cuda().unsqueeze(0)
	data_lowlight = data_lowlight.unsqueeze(0)

	# DCE_net = model.enhance_net_nopool().cuda()
	DCE_net = model.enhance_net_nopool()
	DCE_net.load_state_dict(torch.load('snap/glare.pth', map_location=torch.device('cpu')))
	start = time.time()
	# _,enhanced_image,_,_,_ = DCE_net(data_lowlight)
	_, enhanced_image, _, _ = DCE_net(data_lowlight)
	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('test_data','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'data/test_data/'

		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*")
			for image in test_list:
				# image = image
				print(image)
				lowlight(image)
				# 读取增强后的图像
				enhanced_image_path = image.replace('test_data', 'result')
				enhanced_image = cv2.imread(enhanced_image_path)
				# 对增强后的图像进行去雾处理
				dehazed_image = deHaze(enhanced_image / 255.0)
				# 将去雾后的图像保存
				cv2.imwrite(enhanced_image_path.replace('.jpg', '_dehazed.png'), dehazed_image * 255)




