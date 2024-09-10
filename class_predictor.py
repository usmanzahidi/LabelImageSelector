"""
Started by: Usman Zahidi (uz) {16/02/22}

"""
#general imports
import os, numpy as np, cv2,pickle, logging
from enum                        import Enum,unique
from scipy                       import ndimage

# detectron imports
from detectron2.config           import get_cfg
from detectron2.engine.defaults  import DefaultPredictor
from detectron2                  import model_zoo

# project imports
from detectron2.utils.visualizer        import ColorMode
from fpn_tsne import FPNTSNE

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class ClassPredictor:

    def __init__(self, config_file,metadata_file,num_classes=2, scale=1.0, instance_mode=ColorMode.SEGMENTATION):

        self.instance_mode=instance_mode
        self.scale=scale
        self.metadata=self.get_metadata(metadata_file)
        cfg = self.init_config( config_file, num_classes)

        try:
            self.predictor=DefaultPredictor(cfg)
        except Exception as e:
            logging.error(e)

    def init_config(self, config_file, num_classes=1):
        cfg = get_cfg()
        try:
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
        except Exception as e:
            logging.error(e)


        #cfg.MODEL.WEIGHTS = os.path.join(model_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # 1.strawberry (only strawberry coming from model)
        return cfg

    def get_metadata(self,metadata_file):

        #metadata file has name of classes, it is created to avoid having custom dataset and taking definitions
        # from annotations, instead. It has structure of MetaDataCatlog output of detectron2
        try:
            file = open(metadata_file, 'rb')
        except Exception as e:
            logging.error(e)

        data = pickle.load(file)
        file.close()
        return data

    def get_predictions(self,rgbd_image):
        fpnTSNE = FPNTSNE()
        rgb_image=rgbd_image[:, :, :3]
        fpn_list=list()
        outputs = self.predictor(rgb_image)
        fpn_list.append({'p2':np.around(fpnTSNE.scale_to_01_range(outputs['p2'][0, 0, :, :].cpu().detach().numpy().astype(np.int_))*255),
            'p3':np.around(fpnTSNE.scale_to_01_range(outputs['p3'][0, 0, :, :].cpu().detach().numpy().astype(np.int_))*255),
            'p4':np.around(fpnTSNE.scale_to_01_range(outputs['p4'][0, 0, :, :].cpu().detach().numpy().astype(np.int_))*255),
            'p5':np.around(fpnTSNE.scale_to_01_range(outputs['p5'][0, 0, :, :].cpu().detach().numpy().astype(np.int_))*255),
            'p6':np.around(fpnTSNE.scale_to_01_range(outputs['p6'][0, 0, :, :].cpu().detach().numpy().astype(np.int_))*255),})
        return fpn_list



    def get_background(self,depth_image):

        depth_zero = (depth_image == 0)*1
        depth_zero = depth_zero * np.amax(depth_image)
        depth_image = depth_image + depth_zero
        bg_mask = depth_image > 5
        return bg_mask




    def get_canopy(self,rgb_image):

        I = np.asarray(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV))
        red_range   = [40,122]
        green_range = [21,223]
        blue_range  = [0,223]

        canopy_mask=(I[:,:, 0] >= red_range[0] ) & (I[:,:, 0] <= red_range[1]) & \
        (I[:,:, 1] >= green_range[0] ) & (I[:,:, 1] <= green_range[1]) & \
        (I[:,:, 2] >= blue_range[0] ) & (I[:,:, 2] <= blue_range[1])


        return canopy_mask

    def smooth_seg(self,input_mask,class_name):
        h, w = input_mask.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        input_mask=input_mask*255
        mask=input_mask.astype('uint8')
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        if class_name==ClassNames.CANOPY:
            cv2.floodFill(mask, flood_mask, (0, 0), 255)
            flood_mask=flood_mask[:-2,:-2]
        else:
            flood_mask=mask
        flood_mask=ndimage.binary_fill_holes(flood_mask,structure=np.ones((5,5)))
        return flood_mask

    def get_masks(self,fg_masks, depth_image, class_list):

        # input three foreground class' masks and calculate leftover as background mask
        # then output requested depth masks as per class_list order


        depth_masks=list()
        for classes in class_list:
            if   classes==ClassNames.STRAWBERRY:
                depth_masks.append(fg_masks[:,:,0]*depth_image)
            elif classes == ClassNames.CANOPY:
                depth_masks.append(fg_masks[:,:,1]*depth_image)
            elif classes == ClassNames.RIGID_STRUCT:
                depth_masks.append(fg_masks[:,:,2]*depth_image)
            elif classes == ClassNames.BACKGROUND:
                depth_masks.append(fg_masks[:,:,3]*depth_image)
        return (np.dstack(depth_masks))

@unique
class ClassNames(Enum):
    """
    Enum of different class names
    """

    STRAWBERRY   = 1
    """
    Class strawberry, depicted by yellow colour
    """
    CANOPY       = 2
    """
    Class canopy, depicted by green colour
    """
    RIGID_STRUCT = 3
    """
    Class rigid structure, depicted by red colour
    """

    BACKGROUND   = 4
    """
    Class background, depicted by blue colour
    """



