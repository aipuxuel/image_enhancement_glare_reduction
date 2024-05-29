import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import pytorch_colors as colors
import numpy as np


def bezier_curve(t,p1, p2):
    """
    计算三次贝塞尔曲线上的点
    """
    y = (1 - t) ** 3 * 0 + 3 * t * (1 - t) ** 2 * p1 + 3 * t ** 2 * (1 - t) * p2 + t ** 3
    return y
def bezier_curve_integral(t, p1, p2):
    """
    计算贝塞尔曲线的被积函数
    """
    x = bezier_curve(t, p1, p2)
    return x
def integral(p1,p2):
	x = 0.25*(p1+p2+1)
	return x
class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		# 输入为[3,256,256]
		number_f = 32
		self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)  # [3,256,256]
		self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
		self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
		self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
		self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
		self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
		self.e_conv7 = nn.Conv2d(number_f * 2, 18, 3, 1, 1, bias=True)
		self.e_conv8 = nn.Conv2d(number_f * 2, 6, 3, 1, 1, bias=True)
		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

	def forward(self, x):
		x1 = self.relu(self.e_conv1(x))  # [3,256,256]
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))  # [32,256,256]
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))  # [32,256,256]
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))  # [32,256,256]

		x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))  # [32,256,256]
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))  # [32,256,256]
		x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))  # [18,256,256] 图像增强
		x_a = torch.sigmoid(self.e_conv8(torch.cat([x1, x6], 1))) # [6,256,256] 眩光减弱
		r1, r2, r3, r4, r5, r6 = torch.split(x_r, 3, dim=1)  # 分成6个[3,256,256]
		r7, r8 = torch.split(x_a, 3, dim=1) # 分成2个[3,256,256]

		# 迭代
		x = x + r1 * (torch.pow(x, 2) - x)
		x = x + r2 * (torch.pow(x, 2) - x)
		x = x + r3 * (torch.pow(x, 2) - x)
		x = x + r4*(torch.pow(x,2)-x)  # 迭代4次
		x = x + r5 * (torch.pow(x, 2) - x)
		enhance_image_1 = x + r6 * (torch.pow(x, 2) - x)
		# x = bezier_curve(enhance_image_1, 0.65, r7)
		# integral1 = integral(0.65, r7)
		# enhance_image = x + r8*(torch.pow(x,2)-x) #迭代8次
		x = enhance_image_1 + r7 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
		enhance_image = bezier_curve(x, 0.7, r8)
		integral2 = integral(0.7, r8)
		r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)

		# return enhance_image_1, enhance_image, r, integral1, integral2

		return enhance_image_1, enhance_image, r, integral2






