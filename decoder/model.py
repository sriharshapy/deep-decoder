import torch
import torch.nn as nn
from torchvision import transforms
import copy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


class DeepDecoder(nn.Module):
    def __init__(
        self,
        num_output_channels = 3,
        num_channels_up = [128]*5,
        filter_size_up = 1,
        activation_function = nn.ReLU(),
        need_sigmoid = True,
        pad = 'reflection',
        upsample_first = True,
        upsample_mode = 'bilinear',
        bn_before_act = False,
        bn_affine = True):

        super(DeepDecoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        '''
        Adding last two layers to the decoder with same last layer size given as input
        Example : if input is [128,128] it is transformed to [128,128,128,128]
        '''
        self.num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
        # layers
        self.n_scales = len(self.num_channels_up)

        '''
        Need a value for each layer which must have length of .
        Example: [3,3,3]
        '''
        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
          self.filter_size_up = [filter_size_up] * self.n_scales

        self.layers = nn.Sequential()
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        for i in range(len(self.num_channels_up)-1):

            if upsample_first:
                self.layers.append(self.conv( self.num_channels_up[i], self.num_channels_up[i+1],  self.filter_size_up[i], 1, pad=pad))
                if upsample_mode!='none' and i != len(self.num_channels_up)-2:
                    self.layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
            else:
                if upsample_mode!='none' and i!=0:
                    self.layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
                self.layers.append(self.conv( self.num_channels_up[i], self.num_channels_up[i+1],  self.filter_size_up[i], 1, pad=pad))

            if i != len(self.num_channels_up)-1:
                if(bn_before_act):
                    self.layers.append(nn.BatchNorm2d( self.num_channels_up[i+1] ,affine=bn_affine))
                self.layers.append(activation_function)
                if(not bn_before_act):
                    self.layers.append(nn.BatchNorm2d( self.num_channels_up[i+1], affine=bn_affine))

        self.layers.append(self.conv( self.num_channels_up[-1], num_output_channels, 1, pad=pad))
        if need_sigmoid:
            self.layers.append(nn.Sigmoid())

    def exp_lr_scheduler(self, optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.65**(epoch // lr_decay_epoch))

        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    def fit(
        self,
        img_noisy_var,
        num_channels,
        img_clean_var,
        num_iter=5000,
        LR=0.01,
        OPTIMIZER='adam',
        opt_input=False,
        reg_noise_std=0,
        reg_noise_decayevery=100000,
        mask_var=None,
        apply_f=None,
        lr_decay_epoch=0,
        net_input=None,
        net_input_gen="random",
        find_best=False,
        weight_decay=0,
    ):
      net = self.to(self.device)

      img_noisy_var = img_noisy_var.to(self.device)
      img_clean_var = img_clean_var.to(self.device)

      if net_input != None:
        print("input provided")
      else:
        totalupsample = 2**len(num_channels)
        width = int(img_clean_var.data.shape[2]/totalupsample)
        height = int(img_clean_var.data.shape[3]/totalupsample)
        shape = [1,num_channels[0], width, height]
        print("shape: ", shape)
        net_input = torch.zeros(shape)
        net_input.data.uniform_()
        net_input.data *= 1./10

      net_input = net_input.to(self.device)
      net_input_saved = net_input.clone().to(self.device)
      noise = net_input.clone().to(self.device)

      p = [x for x in net.parameters() ]

      if(opt_input == True): # optimizer over the input as well
        net_input.requires_grad = True
        p += [net_input]

      mse_wrt_noisy = np.zeros(num_iter)
      mse_wrt_truth = np.zeros(num_iter)

      if OPTIMIZER == 'SGD':
          print("optimize with SGD", LR)
          optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9,weight_decay=weight_decay)
      elif OPTIMIZER == 'adam':
          print("optimize with adam", LR)
          optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)
      elif OPTIMIZER == 'LBFGS':
          print("optimize with LBFGS", LR)
          optimizer = torch.optim.LBFGS(p, lr=LR)

      mse = torch.nn.MSELoss() #.type(dtype)
      noise_energy = mse(img_noisy_var, img_clean_var)

      if find_best:
          best_net = copy.deepcopy(net)
          best_mse = 1000000.0
      pbar = tqdm(range(num_iter), desc='Starting training...')
      for i in pbar:
          if lr_decay_epoch != 0:
              optimizer = self.exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)
          if reg_noise_std > 0:
              if i % reg_noise_decayevery == 0:
                  reg_noise_std *= 0.7
              net_input = net_input_saved + (noise.normal_() * reg_noise_std)

          def closure():
              optimizer.zero_grad()
              out = net(net_input)
              # training loss
              if mask_var != None:
                  loss = mse( out * mask_var , img_noisy_var * mask_var )
              elif apply_f:
                  loss = mse( apply_f(out) , img_noisy_var )
              else:
                  loss = mse(out, img_noisy_var)

              loss.backward()
              mse_wrt_noisy[i] = loss.item()

              # the actual loss
              true_loss = mse(out.detach(), img_clean_var)
              mse_wrt_truth[i] = true_loss.item()
              if i % 10 == 0:
                  out2 = net(net_input_saved)
                  loss2 = mse(out2, img_clean_var)
                  #print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f  Noise Energy %f' % (i, loss.item(),true_loss.item(),loss2.item(),noise_energy.item()), '\r', end='')
                  pbar.set_description('T-loss: %.6f | Act-loss: %.6f | Noise Eneg: %.6f' % (loss.item(), true_loss.item(), noise_energy.item()))
              return loss

          loss = optimizer.step(closure)

          if find_best:
              # if training loss improves by at least 0.5 percent, we found a new best net
              if best_mse > 1.005*loss.data:
                  best_mse = loss.data
                  best_net = copy.deepcopy(net)

      if find_best:
          net = best_net

      return mse_wrt_noisy, mse_wrt_truth,net_input_saved, net

    def conv(self, in_f, out_f, kernel_size, stride=1, pad='zero'):
        padder = None
        to_pad = int((kernel_size - 1) / 2)
        if pad == 'reflection':
            padder = nn.ReflectionPad2d(to_pad)
            to_pad = 0

        convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

        layers = filter(lambda x: x != None, [padder, convolver])
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
