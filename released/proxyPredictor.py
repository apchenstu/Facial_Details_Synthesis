import os,cv2,argparse,shutil
import numpy as np


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
    img = cv2.imread(img_path,0)
    if img is None:
        print(img_path)
    im = cv2.resize(img, target_size)
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
        dist = np.sum(abs(dictionaryFAC[:, feature_use] - features[feature_use]), axis=1)+np.sum(abs(dictionaryFAC[:, 2:5] - features[2:5]), axis=1)*10
        
    if 'emotion' == type:
        global dictionaryEmotion
        dist = np.sum(abs(dictionaryEmotion - features), axis=1)

    index = np.argsort(dist)
    result = BFMexp[index[:k],:]
    #print(index[0],result)
    if k > 1:
        result = np.mean(result,axis=0)
    np.savetxt(save_expPath,result, delimiter=' ')

def fit_BFMexp(landmark_path,featurepath,save_exp_path,ptsPath):

    print('Using expression piror from Facial Action Coding features.\n')
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

    images = list_images(args.img_path)
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

        img = cv2.imread(img_name)
        landmark_scale = max(img.shape[0], img.shape[1])/256.0
        save_obj_path = os.path.join(args.output_path, base_name[0], 'result')
        fit_model(args.fitmodel_exe_path, img_name, pts_path, save_obj_path, landmark_scale, save_exp_path)
        #render_texture(args.render_exe_path, img_name, save_obj_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--img_path',required=True,help='image path or images save root')
    parser.add_argument('-o', '--output_path', required=True,help='features file path')
    parser.add_argument('--fit_expression', default=True,help='initial or fit expression from expression or just use as inital')
    parser.add_argument('--emotion', default=False, type=bool, help='whether use emotion features')
    parser.add_argument('--FAC', default=False, type=bool, help='whether use facial action coding features')
    parser.add_argument('-W','--width', default=64)
    parser.add_argument('-H', '--height', default=64)
    parser.add_argument('--write_all',default=False,type=bool,help='whether to write all features together')
    parser.add_argument('--json_file',default='./emotion/checkpoints/large.json',help='path to json file')
    parser.add_argument('--model_file',default='./emotion/checkpoints/large.h5', help='path to pre-trained model')
    parser.add_argument('--landmark_exe_path', default='./landmarks', type=str)
    parser.add_argument('--fitmodel_exe_path', default='./proxy', type=str)
    parser.add_argument('--render_exe_path', default='./renderTexture', type=str)

    args = parser.parse_args()
    main(args)