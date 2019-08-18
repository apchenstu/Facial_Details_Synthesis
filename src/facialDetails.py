import os,shutil,torch
import numpy as np
import scipy.io as io

from PIL import Image
from scipy import signal
from skimage.color import rgb2hsv
from DFDN.options.testsingle_options import TestSingle
from DFDN.models.models import create_model

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def list_images(dir):
    images = []
    if os.path.isdir(dir):
        files = os.listdir(dir)
        for fname in files:
            if is_image_file(fname):
                images.append(os.path.join(dir,fname))
    else:
        if os.path.exists(dir) and is_image_file(dir):
            images.append(dir)
        else:
            print('Please input a valiable image path or images root.')
            exit()
    return sorted(images)

def buildPath(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_image(img_path, target_size):
    img = Image.load(img_path).convert('LA')
    if img is None:
        print(img_path)
    im = np.asarray(img.resize(img, target_size,, Image.ANTIALIAS))
    return im[...,np.newaxis]/255.0*2-1


def inferanceOneImg(args, img_path,feature_model):
    img = load_image(img_path,(args.width,args.height))
    img = img[np.newaxis]  # input should be n*w*h*C
    return feature_model.predict(img).reshape((-1))

def move_landmark(landmark_exe_path,save_path,name):
    shutil.move(os.path.join(landmark_exe_path,'processed',name+'.box'),os.path.join(save_path,name+'.box'))
    shutil.move(os.path.join(landmark_exe_path,'processed',name+'.pts'),os.path.join(save_path,name+'.pts'))
    shutil.move(os.path.join(landmark_exe_path,'processed',name+'.txt'),os.path.join(save_path,name+'.txt'))

def fit_landmarks(landmark_exe_path,image_path):
    current_path = os.getcwd()
    os.chdir(landmark_exe_path)
    if os.path.isfile(image_path):
        cmd = 'FaceLandmarkImg.exe -f ' + image_path
    else:
        cmd = 'FaceLandmarkImg.exe -fdir ' + image_path
    os.system(cmd)
    os.chdir(current_path)

def find_nearest(save_expPath,features,type,k=1):
    global BFMexp
    if 'FAC' == type:
        global dictionaryFAC
        feature_use = range(5,21)
        dist = np.sum(abs(dictionaryFAC[:, feature_use] - features[feature_use]), axis=1)#+np.sum(abs(dictionaryFAC[:, 2:5] - features[2:5]), axis=1)*10
        
    if 'emotion' == type:
        global dictionaryEmotion
        dist = np.sum(abs(dictionaryEmotion - features), axis=1)

    index = np.argsort(dist)
    #print(index[0])
    result = BFMexp[index[:k],:]
    if k > 1:
        result = np.mean(result,axis=0)
    np.savetxt(save_expPath,result, delimiter=' ')

def fit_BFMexp(landmark_path,featurepath,save_exp_path,ptsPath):

    print('Using expression piror from Facial Action Coding features.')
    expression_parameter = np.loadtxt(featurepath)
    find_nearest(save_exp_path,expression_parameter,'FAC',k=1)

def fit_model( exepath, imagePath, ptsPath, savePath,landmark_scale,expressionPath):
    current_path = os.getcwd()
    os.chdir(exepath)

    file_path = ' -p ./bfm2017/ibug_to_bfm2017-1_bfm_nomouth.txt -c ./bfm2017/bfm2017-1_bfm_nomouth_model_contours.json'+ \
				' -e ./bfm2017/bfm2017-1_bfm_nomouth_edge_topology.json -m ./bfm2017/bfm2017-1_bfm_nomouth.bin'
    if expressionPath is None:
        cmd = 'fit-model.exe -i '+ imagePath+' -l '+ptsPath+' -o '+savePath+' --save-texture 0 --save-wireframe 0 --landmark-scale '+ \
		  				str(landmark_scale)+' '+file_path
    else:
        cmd = 'fit-model.exe -i ' +imagePath +' -l ' +ptsPath +' -o ' +savePath+ \
			   ' --num-fit-iter 10 --init-expression-coeffs-fp '+ expressionPath+ \
				' --fix-expression-coeffs ' + str(args.fit_expression) + ' --save-texture 0 --save-wireframe 0  --landmark-scale '+str(landmark_scale) +' '+ file_path

    os.system(cmd)
    os.chdir(current_path)

def render_texture(exepath,imagePath,savePath):
    current_path = os.getcwd()
    os.chdir(exepath)

    objname = savePath +'.obj'
    savename = savePath +'.isomap.png'
    camera = savePath
    cmd = 'faceClip.exe ' +objname +' 3 ' +savename +' 0 ' +imagePath +' ' +camera
    os.system(cmd)
    os.chdir(current_path)

######################################  facial details  ##################################
def crop(img, areas,args):
    patch = np.zeros((areas.shape[1], 1, args.pacthW, args.pacthW))
    for i in range(patch.shape[0]):
        temp = img[areas[0, i]:areas[0, i] + args.pacthW, areas[1, i]:areas[1, i] + args.pacthW]
        patch[i, 0] = (temp - np.mean(temp)) / 127.0
    return torch.FloatTensor(patch)


def loadimage(dataroot, args):
    img = Image.resize(Image.open(dataroot), (args.imageW, args.imageW))
    hsv, gray = rgb2hsv(img), img.convert('LA')
    img = np.array(hsv)[...,2]

    areas = io.loadmat('./DFDN/areas.mat')
    rec = {'forehead': areas['forehead'], 'mouse': areas['mouse'], 'weight': areas['weight']}
    forehead, mouse = crop(img, rec['forehead'],args), crop(img, rec['mouse'],args)

    return {'forehead': forehead, 'mouse': mouse}, rec


def stitch(results, areas, weight, img, count, ind, args):
    for i in range(len(ind)):
        img[areas[0, ind[i]]:areas[0, ind[i]] + args.pacthW, areas[1, ind[i]]:areas[1, ind[i]] + args.pacthW] += results[i, 0] * weight
        count[areas[0, ind[i]]:areas[0, ind[i]] + args.pacthW, areas[1, ind[i]]:areas[1, ind[i]] + args.pacthW] += weight


def calNormalMap(img):
    size, width = 11, img.shape[0]
    kernel_x, kernel_y = np.ones((size, size)), np.ones((size, size))
    kernel_x[:, int((size - 1) / 2)], kernel_x[:, :int((size - 1) / 2)] = 0, -1
    kernel_y[int((size - 1) / 2), :], kernel_y[:int((size - 1) / 2), :] = 0, -1

    GX = signal.fftconvolve(img, kernel_x, 'same')
    GY = signal.fftconvolve(img, kernel_y, 'same')

    scale = 0.07
    normal = np.ones((width, width, 3))
    normal[:, :, 0], normal[:, :, 1] = GX * scale, GY * scale
    len = np.sqrt(np.sum(np.power(normal, 2), 2))
    normal = normal / np.repeat(np.reshape(len, (width, width, 1)), 3, 2)
    return normal


def predict_details(image_root,args):
    global DFDN

    testset, areas = loadimage(image_root,args)
    img, img_hig = np.zeros((args.imageW, args.imageW)), np.zeros((args.imageW, args.imageW))

    with torch.no_grad():
        count = np.zeros((args.imageW, args.imageW))
        for key in DFDN.keys():
            length = testset[key].size(0)
            batch = int(length / args.batchSize + 1)

            for j in range(batch):
                if j != batch - 1:
                    ind = range(args.batchSize * j, args.batchSize * (j + 1))
                else:
                    ind = range(args.batchSize * j, length)
                DFDN[key].input_A = testset[key][ind].to('cuda')
                if 1 == length - args.batchSize * j:
                    DFDN[key].input_A = DFDN[key].input_A.view(1, 1, args.inSize, args.inSize)

                DFDN[key].forward()
                results = DFDN[key].fake_B.cpu().data.numpy()

                stitch(results, areas[key], areas['weight'], img, count, ind,args)

        mask = count > 0
        img[mask] /= count[mask]

        normal = calNormalMap(img)
        return img, normal

DFDN = None
BFMexp,dictionaryFAC,dictionaryEmotion = None,None,None

def main(args):
    global BFMexp,dictionaryFAC,dictionaryEmotion
    if args.emotion:
        from keras.models import Model
        from keras.models import model_from_json
        json_file = open(args.json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(args.model_file)
        output = loaded_model.get_layer('dropout_7').output
        feature_model = Model(inputs=loaded_model.input, outputs=output)
        print("Loaded emotion model from disk")

    if args.FAC or args.emotion:
        diction = np.load(os.path.join(args.landmark_exe_path,'dictionary.npz'))
        BFMexp = diction['expParams']
        if args.FAC:
            dictionaryFAC = diction['expFeatures']
        if args.emotion:
            dictionaryEmotion = diction['emotion']

    args.img_path = os.path.abspath(args.img_path)
    args.output_path = os.path.abspath(args.output_path)
    buildPath(args.output_path)


    # load models
    global DFDN
    args.name, args.dataroot = args.foreheadModel, args.foreheadData
    DFDN = {'forehead': create_model(args)}
    args.name, args.dataroot = args.mouseModel, args.mouseData
    DFDN['mouse'] = create_model(args)


    images = list_images(args.img_path)
    if len(images)==1:
        args.visualize = True

    fit_landmarks(args.landmark_exe_path, args.img_path)
    print('===> Landmarks detection done.\n')
    for img_name in images:
        print('===> estimating proxy of %s '%img_name)
        basename = os.path.basename(img_name)
        base_name = os.path.splitext(basename)

        save_path = os.path.join(args.output_path, base_name[0])
        buildPath(save_path)

        ###################  landmarks  FAC features  ########################
        move_landmark(args.landmark_exe_path,save_path,base_name[0])
        pts_path = os.path.join(save_path, base_name[0] + '.pts')


        #####################   expression pirors  ############################
        save_exp_path = os.path.join(save_path,'expression.txt')
        if args.FAC and not args.emotion:
            feature_path = os.path.join(save_path,base_name[0]+'.txt')
            fit_BFMexp(args.landmark_exe_path,feature_path,save_exp_path,pts_path)
        elif not args.FAC and args.emotion:
            print('Using expression piror from emotion features.\n')
            features_emotion = inferanceOneImg(args, img_name, feature_model)
            find_nearest(save_exp_path,features_emotion,'emotion')
        else:
            save_exp_path = None

        img = Image.open(img_name)
        landmark_scale = max(img.size[0], img.size[1])/256.0

        save_obj_path = os.path.join(args.output_path, base_name[0], 'result')
        fit_model(args.fitmodel_exe_path, img_name, pts_path, save_obj_path, landmark_scale, save_exp_path)
        render_texture(args.render_exe_path, img_name, save_obj_path)


        #######################  predict details  ##################################
        print('===> predicting details of %s '%img_name)
        save_texture_path = os.path.join(args.output_path, base_name[0], 'result.isomap.png')
        displacementMap, normalMap = predict_details(save_texture_path, args)

        displacementMap = (displacementMap+1)/2*65535
        save_path = os.path.join(args.output_path, base_name[0], 'result.displacementmap.png')
        Image.fromarray(displacementMap.astype('uint16')).save(save_path)

        normalMap = (normalMap+1)/2*255
        save_path = os.path.join(args.output_path, base_name[0], 'result.normalmap.png')
        normalMap = normalMap.astype('uint8')
        Image.fromarray(normalMap).save(save_path)

        if args.visualize:
            args.face_render_path = os.path.abspath(args.face_render_path)
            cmd = '%s/hmrenderer.exe %s %s %s'%(args.face_render_path,save_obj_path+'.obj',save_path,args.face_render_path+'/shaders')
            os.system(cmd)
        print('\n')

if __name__ == '__main__':
    args = TestSingle().parse()
    main(args)