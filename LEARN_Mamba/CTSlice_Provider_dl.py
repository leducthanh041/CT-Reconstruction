import os
import numpy as np
import torch
import pydicom
import random
from torchvision.transforms import ToPILImage, ToTensor

from PIL import Image
from glob import glob
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
import os
import math
from odl.contrib import torch as odl_torch
import odl

from odl.tomo.backends import (
    ASTRA_AVAILABLE, ASTRA_CUDA_AVAILABLE, SKIMAGE_AVAILABLE)

#os.environ['RAY_TRAFO_IMPLS'] = 'astra_cuda'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from skimage.io import imread
read_hu = lambda x: imread(x).astype(np.float32)-32768

class CTSlice_Provider(Dataset):

  def __init__(self, base_path,
                    setting: str, poission_level=5e5, gaussian_level=0.05, 
                    num_view=64, transform = None, valid = False, test = False, 
                    input_size =256, num_select = -1):
    self.base_path=base_path
    #self.slices_path=glob(os.path.join(self.base_path,'*/*.dcm'))
    # patients_training = ["L067","L096","L109","L143","L192","L286","L291","L506"]
    # patients_validation = ["L333"]
    # patients_test = ["L310"]
    
    # paths_training = []
    # paths_test = []
    # self.input_size = input_size

    self.sino_path = str(self.base_path) + "train/" + setting + "/sino"
    self.fbp_u_path = str(self.base_path) + "train/" + setting + "/fbp_u"

    if valid:
      val_list = [path for path in open(str(self.base_path)+'val.csv').read().split('\n')]         
      self.slices_path = val_list
  
    elif test:
      test_list = [path for path in open(str(self.base_path)+'test.csv').read().split('\n')]
      self.slices_path = test_list

    else:
      train_list = [path for path in open(str(self.base_path)+'train.csv').read().split('\n')]
      self.slices_path = train_list

    self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full=self._radon_transform(num_view=360)
    self.radon_curr, self.iradon_curr, self.fbp_curr, self.op_norm_curr=self._radon_transform(num_view=num_view)


    self.poission_level=poission_level
    self.gaussian_level=gaussian_level
    self.num_view=num_view
    self.transform = transform

    print("poission_level", self.poission_level)

  def _radon_transform(self, num_view= 96, start_ang=0, end_ang=2*np.pi, num_detectors=512):
    # the function is used to generate fp, bp, fbp functions
    # the physical parameters is set as MetaInvNet and EPNet
    xx=200
    space=odl.uniform_discr([-xx, -xx], [xx, xx], [256,256], dtype='float32')
    angles=np.array(num_view).astype(int)
    angle_partition = odl.uniform_partition(start_ang, end_ang, angles)
    detector_partition=odl.uniform_partition(-480, 480, num_detectors)
         
    geometry=odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
    #geometry=odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
    operator=odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm=odl.operator.power_method_opnorm(operator)
    op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()
    
    op_layer=odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint=odl_torch.operator.OperatorModule(operator.adjoint)
    fbp=odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9)*np.sqrt(2)
    op_layer_fbp=odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

  def __getitem__(self, index):  
    slice_path=self.slices_path[index]
    data_slice = np.asarray(read_hu(slice_path))
    data_slice=np.array(data_slice).astype(float)
    data_slice=(data_slice-np.min(data_slice))/(np.max(data_slice)-np.min(data_slice))

    # the following code is used to generate projections with noise in the way of odl package
    phantom=torch.from_numpy(data_slice).unsqueeze(0).type(torch.FloatTensor)
    sino=np.load(self.sino_path + "/" + '/'.join(slice_path.split('/')[-2:]).split(".png")[0] + ".npy")
    sino=torch.Tensor(sino)
    # fbp_u=np.load(self.fbp_u_path + "/" + os.path.basename(slice_path).split(".IMA")[0] + ".npy")
    
    if self.poission_level > 0:
      # add poission noise
      intensityI0=self.poission_level
      scale_value=torch.from_numpy(np.array(intensityI0).astype(float))
      normalized_sino=torch.exp(-sino/sino.max())
      th_data=np.random.poisson(scale_value*normalized_sino)
      sino_noisy=-torch.log(torch.from_numpy(th_data)/scale_value)
      sino_noisy = sino_noisy*sino.max()
      
      # add Gaussian noise
      noise_std=self.gaussian_level
      noise_std=np.array(noise_std).astype(float)
      nx,ny=np.array(self.num_view).astype(int), np.array(sino.shape[-1]).astype(int)
      noise = noise_std*np.random.randn(nx,ny)
      noise = torch.from_numpy(noise)
      sino_noisy = sino_noisy + noise
      fbp_u=self.fbp_curr(sino_noisy)

    else:
      sino_noisy = sino
      fbp_u=np.load(self.fbp_u_path + "/" + '/'.join(slice_path.split('/')[-2:]).split(".png")[0] + ".npy")     

    if self.transform:
      # phantom = self.transform(phantom)
      '''NEW'''
      # Chuyển từ tensor sang PIL Image
      pil_img = ToPILImage()(phantom)
      # Áp dụng transform
      pil_img = self.transform(pil_img)
      # Chuyển ngược lại thành tensor nếu cần
      phantom = ToTensor()(pil_img)
      '''NEW'''
    
    return phantom, fbp_u, sino_noisy

  def __len__(self):
    return len(self.slices_path)