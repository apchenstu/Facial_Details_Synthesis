from .base_options import BaseOptions


class TestSingle(BaseOptions):
    def initialize(self):
        self.parser.add_argument('-i','--img_path',required=True,help='image path or images save root')
        self.parser.add_argument('-o', '--output_path', required=True,help='features file path')
        self.parser.add_argument('--fit_expression', default=True,help='initial or fit expression from expression or just use as inital')
        self.parser.add_argument('--emotion', default=False, type=bool, help='whether use emotion features')
        self.parser.add_argument('--FAC', default=True, type=bool, help='whether use facial action coding features')
        self.parser.add_argument('-W','--width', default=64)
        self.parser.add_argument('-H', '--height', default=64)
        self.parser.add_argument('--write_all',default=False,type=bool,help='whether to write all features together')
        self.parser.add_argument('--json_file',default='./emotion/checkpoints/large.json',help='path to json file')
        self.parser.add_argument('--model_file',default='./emotion/checkpoints/large.h5', help='path to pre-trained model')
        self.parser.add_argument('--landmark_exe_path', default='./landmarks', type=str)
        self.parser.add_argument('--fitmodel_exe_path', default='./proxy', type=str)
        self.parser.add_argument('--render_exe_path', default='./renderTexture', type=str)
        self.parser.add_argument('--details_exe_path', default='./DFDN', type=str)
        self.parser.add_argument('--face_render_path', default='./faceRender', type=str)
        self.parser.add_argument('--visualize', default=False,type=bool)

        ############################### DFDN PARSERS ######################################
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=100.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--trainPersent', type=float, default=0.9,help='how many persent of training dataset')

        self.parser.add_argument('--foreheadModel',type=str,default='forehead_1031')
        #self.parser.add_argument('--highfrequencyModel',type=str,default='1111')
        self.parser.add_argument('--mouseModel',type=str,default='mouse_1031')

        self.parser.add_argument('--foreheadData',type=str,default='./DFDN/checkpoints/PCA/forehead/')
        self.parser.add_argument('--mouseData',type=str,default='./DFDN/checkpoints/PCA/mouse/')


        self.parser.add_argument('--imageW', type=int, default=4096,help='how many persent of training dataset')
        self.parser.add_argument('--pacthW', type=int, default=256,help='how many persent of training dataset')
        self.isTrain = False
