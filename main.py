"""
Started by: Usman Zahidi (uz) {16/02/22}

"""
import os
import numpy as np, cv2,argparse,logging
from class_predictor import ClassPredictor
from torch.distributions import Categorical
from fpn_tsne import FPNTSNE
from sklearn.manifold import TSNE
from os import listdir,path
import kmeans_pytorch as kmeans_pt
import torch, shutil, inspect, detectron2


parser = argparse.ArgumentParser(description="Labelling Data Selector")
parser.add_argument("-i", "--image", default='', type=str, metavar="PATH", help="path to image folder")
parser.add_argument("-o", "--output", default='', type=str, metavar="PATH", help="path to output folder")
parser.add_argument("-n", "--no_of_images", default='', type=str, metavar="PATH", help="required number of images")
parser.add_argument('--entropy', default=False, action='store_true')
parser.add_argument('--no-entropy', dest='entropy', action='store_false')
entropy_list=list()

def call_selector():


    args = parser.parse_args()
    assert(not(not args.image or not args.output)), "invalid parameters provided"

    image_dir = args.image + '/'
    output_dir = args.output + '/'
    no_of_images =  int(args.no_of_images)
    include_entropy = args.entropy

    p2_dir   = image_dir + '/tsne/p2/'
    p3_dir   = image_dir + '/tsne/p3/'
    tsne_dir = image_dir + '/tsne/'
    config_file = 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
    metadata_file ='./data/metadata.pkl'
    update_rcnn(tsne_dir,p2_dir,p3_dir)

    rgb_files = [f for f in listdir(image_dir) if path.isfile(image_dir+f)]
    p2_list=list()
    classPred = ClassPredictor(config_file, metadata_file)

    #loop for generating/saving segmentation output images
    k=1
    for rgb_file in rgb_files:
        file_name=image_dir+rgb_file
        rgb_image   = cv2.imread(file_name)

        if rgb_image is None:
            message = 'path to image is invalid'
            logging.error(message)

        fpn=classPred.get_predictions(rgb_image)
        p2_list.append(fpn[0]['p2'])
        p2_list.append(fpn[0]['p2'])
        p2_list.append(fpn[0]['p2'])
        p2_image = np.dstack(p2_list)
        cv2.imwrite(path.join(p2_dir,rgb_file),p2_image)
        p2_list.clear()
        p_tensor = torch.tensor((np.ravel(p2_image)))
        p2_ent = Categorical(probs=p_tensor).entropy()

        p2_list.append(fpn[0]['p3'])
        p2_list.append(fpn[0]['p3'])
        p2_list.append(fpn[0]['p3'])
        p2_image = np.dstack(p2_list)
        cv2.imwrite(path.join(p3_dir, rgb_file), p2_image)
        p2_list.clear()
        p_tensor = torch.tensor((np.ravel(p2_image)))
        p3_ent = Categorical(probs=p_tensor).entropy()
        entropy_list.append([p2_ent,p3_ent])
        print(str(k)+ '/'+str(len(rgb_files))+ ' processed')
        k += 1

    select_images(image_dir,output_dir,tsne_dir,no_of_images,include_entropy)


def update_rcnn(tsne_dir, p2_dir, p3_dir):
    file_path, tail = path.split(inspect.getfile(detectron2))
    file_path = file_path + '/modeling/meta_arch/'
    shutil.copy('./cnns/rcnn.py', file_path)

    if not path.exists(tsne_dir):
        os.mkdir(tsne_dir)
    if not path.exists(p2_dir):
        os.mkdir(p2_dir)
    if not path.exists(p3_dir):
        os.mkdir(p3_dir)

def select_images(image_dir,output_dir,tsne_dir,no_of_images,include_entropy):
    total_no = len(listdir(image_dir))
    fpnTSNE=FPNTSNE()
    fpnTSNE.fix_random_seeds()

    folders = listdir(tsne_dir)
    tsne_list = list()
    index = 0
    for folder in folders:
        file_list = listdir(path.join(tsne_dir, folder))
        num_images = len(file_list)
        features, labels, image_paths = fpnTSNE.get_features(
            dataset=tsne_dir,
            batch=64,
            folder=folder,
            num_images=total_no
        )
        if index == 0:
            full_image_paths = image_paths.copy()
        tsne_list.append(features)
        index += 1

    image_paths = full_image_paths

    if len(tsne_list) >= 2:
        tsne = np.concatenate((tsne_list[0], tsne_list[1]), 1)
        for i in range(2, len(folders)):
            tsne = np.concatenate((tsne, tsne_list[i]), 1)
    else:
        tsne = tsne_list[0]
    tsne = TSNE(n_components=2).fit_transform(tsne)
    if include_entropy:
        tsne = np.concatenate((tsne, np.asarray(entropy_list)), 1)
    x = torch.tensor(tsne)
    num_clusters = no_of_images
    cluster_ids_x, cluster_centers = kmeans_pt.kmeans(
        X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    )
    class_list = fpnTSNE.get_nearest_images(x, cluster_centers)
    image_paths = [image_paths[i] for i in class_list]
    for image_path in image_paths:
        head,tail=path.split(image_path)
        source=image_dir+tail
        shutil.copy(source, output_dir)

if __name__ == '__main__':
    #example call
    #python main.py -i ./images/ -d ./output/
    call_selector()


