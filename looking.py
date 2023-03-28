import os, errno
import argparse
import time

import logging
log = logging.getLogger('LOOKLog')
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s | %(asctime)s | %(name)s | %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

from .utils.network import *
from .utils.utils_predict import *

from PIL import Image, ImageFile

"""
PIFPAF is a command line tool and likes command line arguments, so for ease of use we use ArgParser to pass values to PIFPAF
"""
parser = argparse.ArgumentParser(prog='python3 predict', usage='%(prog)s [options] images', description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=1, type=int, help='processing batch size')
parser.add_argument('--device', default='0', type=str, help='cuda device')
parser.add_argument('--long-edge', default=None, type=int, help='rescale the long side of the image (aspect ratio maintained)')
parser.add_argument('--loader-workers', default=None, type=int, help='number of workers for data loading')
parser.add_argument('--precise-rescaling', dest='fast_rescaling', default=True, action='store_false', help='use more exact image rescaling (requires scipy)')
parser.add_argument('--checkpoint_', default='shufflenetv2k30', type=str, help='backbone model to use')
parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')
parser.add_argument('--time', action='store_true', help='track comptutational time')
""""""

decoder.cli(parser)
logger.cli(parser)
network.Factory.cli(parser)
show.cli(parser)
visualizer.cli(parser)

args = parser.parse_args()

DOWNLOAD = None
INPUT_SIZE=51
FONT = cv2.FONT_HERSHEY_SIMPLEX

ImageFile.LOAD_TRUNCATED_IMAGES = True

log.info('OpenPifPaf version'+openpifpaf.__version__)
log.info('PyTorch version'+torch.__version__)

class Predictor():
    """
        Class definition for the predictor.
        For user customization of the PIFPAF arguments, access is given to the PIFPAF args on 
        initialization of the class and are set again if changed from default defined above
    """
    def __init__(self, transparency=0.4, looking_threshold=0.5, mode='joints', device=args.device, pifpaf_ver='shufflenetv2k30', model_path='models/predictor',
                batch_size=args.batch_size, long_edge=args.long_edge, loader_workers=args.loader_workers, disable_cuda=args.disable_cuda):
        self.looking_threshold = looking_threshold
        self.transparency = transparency
        self.mode = mode

        #PIFPAF ARGS
        args.checkpoint = pifpaf_ver
        args.force_complete_pose = True
        args.batch_size=batch_size
        args.long_edge=long_edge
        args.load_workers=loader_workers
        args.disable_cuda=disable_cuda
        args.force_complete_pose = True
        #PIFPAF ARGS

        if device != 'cpu':
            use_cuda = torch.cuda.is_available()
            #print(torch.cuda.is_available())
            self.device = torch.device("cuda:{}".format(device) if use_cuda else "cpu")
        else:
            self.device = torch.device('cpu')
        args.device = self.device
        log.info(f'Device being used: {self.device}')
        self.predictor_ = load_pifpaf(args)
        self.path_model = model_path
        try:
            os.makedirs(self.path_model)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.model = self.get_model().to(self.device)
        self.track_time = args.time
        if self.track_time:
            self.pifpaf_time = []
            self.inference_time = []
            self.total_time = []

        self.pred_labels=None
        self.pifpaf_outs=None
        self.output_image=None
        self.boxes=None
        self.keypoints=None

    
    def get_model(self):
        """Get Model
        Internal function to get the correcdt model to use
        Arguments:
        Returns:
            model
        """
        if self.mode == 'joints':
            model = LookingModel(INPUT_SIZE)
            dir_name=os.path.dirname(__file__)
            if self.path_model=='models/predictor':
                self.path_model=os.path.join(dir_name,'models/predictor')
            if not os.path.isfile(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p')):
                """
                DOWNLOAD(LOOKING_MODEL, os.path.join(self.path_model, 'Looking_Model.zip'), quiet=False)
                with ZipFile(os.path.join(self.path_model, 'Looking_Model.zip'), 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall()
                exit(0)"""
                raise NotImplementedError
            model.load_state_dict(torch.load(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p'), map_location=self.device))
            model.eval()
        else:
            model = AlexNet_head(self.device)
            if not os.path.isfile(os.path.join(self.path_model, 'AlexNet_LOOK.p')):
                """
                DOWNLOAD(LOOKING_MODEL, os.path.join(self.path_model, 'Looking_Model.zip'), quiet=False)
                with ZipFile(os.path.join(self.path_model, 'Looking_Model.zip'), 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall()
                exit(0)"""
                raise NotImplementedError
            model.load_state_dict(torch.load(os.path.join(self.path_model, 'AlexNet_LOOK.p')))
            model.eval()
        return model

    def predict_look(self, boxes, keypoints, im_size, batch_wise=True) -> list:
        """Predict Look
        Internal function to predict looking using keypoints
        Arguments:
            bbox: list , the bounding boxes from PP prediction
            keypoints: list, the keypoints from PP prediction
            im_size: tuple , (w,h) image dimensions
            batch_wise: bool
        Returns:
            out_labels: list
        """
        label_look = []
        final_keypoints = []
        if batch_wise:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
                    kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    #X, Y = normalize(X, Y, divide=True, height_=False)
                    kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                    final_keypoints.append(kps_final_normalized)
                tensor_kps = torch.Tensor([final_keypoints]).to(self.device)
                if self.track_time:
                    start = time.time()
                    out_labels = self.model(tensor_kps.squeeze(0)).detach().cpu().numpy().reshape(-1)
                    end = time.time()
                    self.inference_time.append(end-start)
                else:
                    out_labels = self.model(tensor_kps.squeeze(0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        else:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
                    kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    #X, Y = normalize(X, Y, divide=True, height_=False)
                    kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                    #final_keypoints.append(kps_final_normalized)
                    tensor_kps = torch.Tensor(kps_final_normalized).to(self.device)
                    if self.track_time:
                        start = time.time()
                        out_labels = self.model(tensor_kps.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
                        end = time.time()
                        self.inference_time.append(end-start)
                    else:
                        out_labels = self.model(tensor_kps.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        return out_labels
    
    def predict_look_alexnet(self, boxes, image, batch_wise=True) -> list:
        """Predict Look AlexNet
        Internal function to predict looking using AlexNet
        Arguments:
            bbox: list , the bounding boxes from PP prediction
            image: internal format image
            batch_wise: bool
        Returns:
            out_labels: list
        """
        out_labels = []
        data_transform = transforms.Compose([
                        SquarePad(),
                        transforms.Resize((227,227)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        if len(boxes) != 0:
            if batch_wise:
                heads = []
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    x1, y1, x2, y2, _ = bbox
                    w, h = abs(x2-x1), abs(y2-y1)
                    head_image = Image.fromarray(np.array(image)[int(y1):int(y1+(h/3)), int(x1):int(x2), :])
                    head_tensor = data_transform(head_image)
                    heads.append(head_tensor.detach().cpu().numpy())
                if self.track_time:
                    start = time.time()
                    out_labels = self.model(torch.Tensor([heads]).squeeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)
                    end = time.time()
                    self.inference_time.append(end-start)
            else:
                out_labels = []
                for i in range(len(boxes)):
                    bbox = boxes[i]
                    x1, y1, x2, y2, _ = bbox
                    w, h = abs(x2-x1), abs(y2-y1)
                    head_image = Image.fromarray(np.array(image)[int(y1):int(y1+(h/3)), int(x1):int(x2), :])
                    head_tensor = data_transform(head_image)
                    #heads.append(head_tensor.detach().cpu().numpy())
                    if self.track_time:
                        start = time.time()
                        looking_label = self.model(torch.Tensor(head_tensor).unsqueeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)[0]
                        end = time.time()
                        self.inference_time.append(end-start)
                    else:
                        looking_label = self.model(torch.Tensor(head_tensor).unsqueeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)[0]
                    out_labels.append(looking_label)
                #if self.track_time:
                #    out_labels = self.model(torch.Tensor([heads]).squeeze(0).to(self.device)).detach().cpu().numpy().reshape(-1)
        else:
            out_labels = []
        return out_labels

    def render_image(self, image, bbox, keypoints, pred_labels) -> np.ndarray:
        """Render Image
        Internal function that takes the iternal image format and predictions and outputs 
        the image with overlayed predictions
        Arguments:
            image: internal format image
            bbox: list , the bounding boxes from PP prediction
            keypoints: list , the keypoints from PP prediction
            pred_labels: list , the predictions of looking attention
        Returns:
            open_cv_image: np.ndarray , the opencv format image
        """
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        scale = 0.007
        imageWidth, imageHeight, _ = open_cv_image.shape
        font_scale = min(imageWidth,imageHeight)/(10/scale)

        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)
        for i, label in enumerate(pred_labels):
            if label > self.looking_threshold:
                color = (0,255,0)
            else:
                color = (255,0,0)
            mask = draw_skeleton(mask, keypoints[i], color)
        mask = cv2.erode(mask,(7,7),iterations = 1)
        mask = cv2.GaussianBlur(mask,(3,3),0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, self.transparency, 1.0)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

        return open_cv_image

    def predict(self, image: np.ndarray):
        """Prediction Function
        Arguments:
            image: a RGB image in the form of a numpy array(OpenCV format)
        Returns:
            None
        """
        self.input_image = image

        if(image is None or image.ndim!=3):
            log.error('ERROR: Image is none or of wrong format')
            exit()
        img_width, img_height, c = image.shape
        
        try:
            cpu_image = Image.fromarray(image)

            pp_predictions, pp_anns, pp_meta = self.predictor_.numpy_image(image)

            pifpaf_outs = {
                'json_data' : [ann.json_data() for ann in pp_predictions],
                'image' : cpu_image
            }

            im_size = (cpu_image.size[0], cpu_image.size[1])
            self.boxes, self.keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)
            if self.mode == 'joints':
                self.pred_labels = self.predict_look(self.boxes, self.keypoints, im_size)
            else:
                self.pred_labels = self.predict_look_alexnet(self.boxes, cpu_image)

            self.output_image = self.render_image(pifpaf_outs['image'], self.boxes, self.keypoints, self.pred_labels)

        except Exception as err:
            log.error(f'{err}, {type(err)}')

    def get_pred_labels(self) -> list:
        """Prediction Labels
        Returns:
            list: pred_labels
        """
        return self.pred_labels
    def get_pifpaf_output(self) -> dict:
        """PIFPAF Output Dictionary
        Returns:
            dict: pifpaf_outs
        """
        return self.pifpaf_outs
    def get_output_image(self) -> np.ndarray:
        """OpenCV Output Image
        Returns:
            np.ndarray: output_image
        """
        return self.output_image
    def get_bounding_boxes(self) -> list:
        """Bounding Boxes
        Returns:
            list: boxes
        """
        return self.boxes
    def get_keypoints(self) -> list:
        """Keypoints
        Returns:
            list: keypoints
        """
        return self.keypoints