import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from DFDN.util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import scipy.io as io


class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain and not opt.test
        # define tensors
        
        save_filename = 'loss.log'
        save_path = os.path.join(self.opt.checkpoints_dir,self.opt.name, save_filename)
        self.log = open(save_path,'a')
        
        dim = self.opt.dimention

        self.z = Variable(torch.ones(self.opt.inSize,self.opt.inSize).to('cuda')/1667)
        self.filter_x,self.filter_y = Variable(torch.zeros(1,1,3,3).to('cuda')),Variable(torch.zeros(1,1,3,3).to('cuda'))
        self.filter_x[0,0,0,1],self.filter_x[0,0,2,0] = -0.5,0.5
        self.filter_y[0,0,0,1],self.filter_y[0,0,2,2] = -0.5,0.5

        if self.opt.matroot:
            self.Project = np.array(io.loadmat(self.opt.dataroot+'project.mat')['p'])
            self.Project = torch.cuda.FloatTensor(self.Project[:,:dim])
            self.project = Variable(torch.cuda.FloatTensor(self.Project.reshape((self.opt.inSize,self.opt.inSize,dim))))


        #if input_A.size()[0] > self.opt.fineSize_w:
        if self.opt.inSize > self.opt.fineSize_w:
            
            self.opt.slip = int(self.opt.inSize/self.opt.fineSize_w)
            self.input_A = self.Tensor(opt.batchSize*self.opt.slip*self.opt.slip, opt.input_nc,
                                           opt.fineSize_h, opt.fineSize_w)
            self.input_B = self.Tensor(opt.batchSize*self.opt.slip*self.opt.slip, opt.output_nc,
                                           opt.fineSize_h, opt.fineSize_w)
            
        else:
            self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                           opt.fineSize_h, opt.fineSize_w)
            self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                           opt.fineSize_h, opt.fineSize_w)
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids,False)


        
        if self.opt.reconstructLoss:
                self.netLIGHT = networks.FC_layers(opt.inSize*opt.inSize,4,self.gpu_ids)

        if self.opt.matroot:
            self.netFC = networks.FC_layers(opt.inSize*opt.inSize,dim,self.gpu_ids)
            if self.opt.which_refinemodel == 'unet':
                self.refineCNN = networks.define_G(opt.output_nc*2, opt.output_nc, opt.ngf,
                      'unet_64', opt.norm, opt.use_dropout, self.gpu_ids,False)
            elif self.opt.which_refinemodel == 'resnet':
                self.refineCNN = networks.define_G(opt.output_nc*2, opt.output_nc, opt.ngf,
                                      'resnet_6blocks', opt.norm, opt.use_dropout, self.gpu_ids)
            else:
                self.refineCNN = networks.RefineLayers(opt.output_nc*2, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc+opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.opt.matroot:
                self.load_network(self.netFC, 'FC', opt.which_epoch)
                self.load_network(self.refineCNN, 'REFINE', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
            if self.opt.reconstructLoss:
                self.load_network(self.netLIGHT, 'LIGHT', opt.which_epoch)
            print('Load preTrain model done.')

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),             
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_L = torch.optim.Adam(self.netD.parameters(),             
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            if self.opt.matroot:
                self.optimizer_R = torch.optim.Adam(self.refineCNN.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_F = torch.optim.Adam(self.netFC.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))            

        if self.opt.isTrain:
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            if self.opt.reconstructLoss:
                    networks.print_network(self.netLIGHT)
            if self.opt.matroot:
                networks.print_network(self.netFC)
                networks.print_network(self.refineCNN)
            if self.isTrain:
                networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A,input_B = input['A' if AtoB else 'B'],input['B' if AtoB else 'A']
        if self.opt.reconstructLoss:
            self.posiction = input['P']
        
        #if input_A.size()[2]>self.opt.fineSize_w:
        if input_A.size()[0]>self.opt.fineSize_w:   
                
            w,h,slip = self.opt.fineSize_w,self.opt.fineSize_h,self.opt.slip
            for batchsize in range(self.opt.batchSize):
                for row in range(slip):
                    for col in range(slip):
                        self.input_A[row*slip+col] = input_A[batchsize,:, h*row:h*(row+1),w*col:w*(col+1)]
                        self.input_B[row*slip+col] = input_B[batchsize,:, h*row:h*(row+1),w*col:w*(col+1)]
        else:
            self.input_A.resize_(input_A.size()).copy_(input_A)
            self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_B = Variable(self.input_B)
        if self.opt.matroot:


            self.real_A = Variable(self.input_A)    

            predict = self.netG.forward(self.real_A)
            vectors = predict.view(predict.size()[0],predict.size()[1]*predict.size()[2]*predict.size()[3])
            self.weights = self.netFC.forward(vectors)

           
            Outputsize = list(self.input_A.size())

            Outputsize[1] *= 2
            Outputsize = torch.Size(Outputsize)
            Inputrefine = Variable(torch.cuda.FloatTensor(Outputsize).zero_())
            self.fake_B = Variable(torch.cuda.FloatTensor(self.input_A.size()).zero_())
            

            for i in range(self.input_A.size()[0]):
                Inputrefine[i,0,:,:] = torch.matmul(self.project,self.weights[i,:].view(self.weights[i].size()[0],1)).view(self.input_A.size()[2],self.input_A.size()[3])


            Inputrefine[:,1,:,:] = self.real_A[:,0,:,:]
            self.pcaRecontruct = Variable(torch.cuda.FloatTensor(self.input_A.size()).zero_())
            self.pcaRecontruct[:,0,:,:] = Inputrefine[:,0,:,:]
            Inputrefine.detach()
            fake_B = self.refineCNN.forward(Inputrefine)
            self.fake_B[:,0] = fake_B[:,0]


        else:
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        out = torch.cat((self.real_A.data,self.fake_B.data,self.real_B.data),3).cpu().numpy()
        return OrderedDict([('result',out)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        if self.opt.matroot:
            self.save_network(self.netFC, 'FC', label, self.gpu_ids)
            self.save_network(self.refineCNN, 'REFINE', label, self.gpu_ids)
        if self.opt.reconstructLoss:
            self.save_network(self.netLIGHT, 'LIGHT', label, self.gpu_ids)
        loss = self.get_current_errors()
        self.log.write('==>epoch '+str(label)+': ')
        for iteam in loss:
            self.log.write(str(iteam)+' '+str(loss[iteam])+' ')
        self.log.write('\n')

