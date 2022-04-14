import os
from os import listdir
from tqdm import tqdm
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from os.path import isfile, join
import shutil

from fpn_dataset import FPNDataset, collate_skip_empty, colors_per_class
from cnns.resnet import ResNet101


class FPNTSNE(torch.utils.data.Dataset):
    def __init__(self):
        self.value=0
        #self.data_path = data_path
        #self.folder = folder
        #self.num_images = 1000

    def scale_to_01_range(self,x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    def fix_random_seeds(self):
        seed = 10
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def get_features(self, dataset, batch, folder, num_images):
        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        # initialize our implementation of ResNet
        model = ResNet101(pretrained=True)
        model.eval()
        model.to(device)

        # read the dataset and initialize the data loader
        dataset = FPNDataset(dataset, folder, num_images)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate_skip_empty, shuffle=True)

        # we'll store the features as NumPy array of size num_images x feature_size
        features = None

        # we'll also store the image labels and paths to visualize them later
        labels = []
        image_paths = []

        for batch in tqdm(dataloader, desc='Running the model inference'):
            images = batch['image'].to(device)
            labels += batch['label']
            image_paths += batch['image_path']

            with torch.no_grad():
                output = model.forward(images)

            current_features = output.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features
        return features, labels, image_paths

    # scale and move the coordinates so they fit [0; 1] range

    def scale_image(self, image, max_image_size):
        image_height, image_width, _ = image.shape

        scale = max(1, image_width / max_image_size, image_height / max_image_size)
        image_width = int(image_width / scale)
        image_height = int(image_height / scale)

        image = cv2.resize(image, (image_width, image_height))
        return image

    def draw_rectangle_by_class(self, image, label):
        image_height, image_width, _ = image.shape

        # get the color corresponding to image class
        color = colors_per_class[label]
        image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

        return image

    def compute_plot_coordinates(self, image, x, y, image_centers_area_size=0, offset=0):
        image_height, image_width, _ = image.shape

        # compute the image center coordinates on the plot
        center_x = int(image_centers_area_size * x) + offset

        # in matplotlib, the y axis is directed upward
        # to have the same here, we need to mirror the y coordinate
        center_y = int(image_centers_area_size * (1 - y)) + offset

        # knowing the image center, compute the coordinates of the top left and bottom right corner
        tl_x = center_x - int(image_width / 2)
        tl_y = center_y - int(image_height / 2)

        br_x = tl_x + image_width
        br_y = tl_y + image_height

        return tl_x, tl_y, br_x, br_y

    def visualize_tsne_images(self, tx, ty, images, labels, plot_size=None, max_image_size=500, fig=None, fig_no=None):
        # we'll put the image centers in the central area of the plot
        # and use offsets to make sure the images fit the plot
        offset = max_image_size // 2
        image_centers_area_size = plot_size[0] - 2 * offset

        tsne_plot = 255 * np.ones((plot_size[0], plot_size[1], 3), np.uint8)

        for image_path, label, x, y in tqdm(
                zip(images, labels, tx, ty),
                desc='Building the T-SNE plot',
                total=len(images)
        ):
            image = cv2.imread(image_path)

            # scale the image to put it to the plot
            image = self.scale_image(image, max_image_size)

            # draw a rectangle with a color corresponding to the image class
            # if i == 0 and label == "excluded":
            image = self.draw_rectangle_by_class(image, label)
            # elif i == 1 and label == "selected":
            #    image = draw_rectangle_by_class(image, label)
            # compute the coordinates of the image on the scaled plot visualization
            tl_x, tl_y, br_x, br_y = self.compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

            # put the image to its TSNE coordinates using numpy subarray indices
            tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

        ax = fig.add_subplot(fig_no)
        plt.imshow(tsne_plot[:, :, ::-1])
        plt.show()

    def visualize_tsne_points(self, tx, ty, labels, fig=None, fig_no=None):
        # initialize matplotlib plot
        if fig is None:
            fig = plt.figure()
            fig_no=111
        ax = fig.add_subplot(fig_no)

        # for every class, we'll add a scatter plot separately
        for label in colors_per_class:
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format:
            # BGR -> RGB, divide by 255, convert to np.array
            color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

            # add a scatter plot with the correponding color and label
            ax.scatter(current_tx, current_ty, c=color, label=label)

        # build a legend using the labels we set previously
        ax.legend(loc='best')

        # finally, show the plot
        # plt.show()
        return fig

    def visualize_tsne(self, tsne, images, labels, plot_size=1000, max_image_size=200):
        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = tsne[:, 0]
        ty = tsne[:, 1]

        # scale and move the coordinates so they fit [0; 1] range
        tx = self.scale_to_01_range(tx)
        ty = self.scale_to_01_range(ty)

        # visualize the plot: samples as colored points
        fig = self.visualize_tsne_points(tx, ty, labels)
        fig_no=122
        #file_name = "./scatter.dat"
        #if os.path.isfile(file_name):
        #    with open(file_name, 'rb') as fh:
        #        dat_list = pickle.load(fh)
        #else:
        #    dat_list = list()
        #dat_list.append(tx)
        #dat_list.append(ty)
        #dat_list.append(images)
        #dat_list.append(labels)
        #with open(file_name, 'wb') as fh:
        #    pickle.dump(dat_list, fh)

        # visualize the plot: samples as images
        self.visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size, fig=fig,fig_no=fig_no)

    def delete_files_in_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            files = listdir(folder_path)
            for f in files:
                os.remove(join(folder_path, f))

    def copy_files(self, file_list, folder_path, ):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            self.delete_files_in_folder(folder_path)
        for image_file_name in file_list:
            if isfile(image_file_name):
                shutil.copy(image_file_name, folder_path)

    def copy_random_files(self, file_list, folder_path, sample_no):
        self.delete_files_in_folder(folder_path)
        for image_file_name in random.sample(file_list, sample_no):
            if isfile(image_file_name):
                shutil.copy(image_file_name, folder_path)

    def distance(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    def get_nearest_images(self, xvals, cluster_centers):
        xvals = xvals.cpu().detach().numpy()
        cluster_centers = cluster_centers.cpu().detach().numpy()
        index_list = list()
        dist_list = list()
        image_no = 0
        for centroid in cluster_centers:
            for x in xvals:
                dist_list.append(self.distance(x, centroid))
            index_list.append(dist_list.index(min(dist_list)))
            dist_list.clear()
            image_no += 1
        return np.unique(index_list)

    def mean_tsne(self, tsne_list):
        count = 1
        for tsne in tsne_list:
            tsne = self.scale_to_01_range(tsne)
            if count == 1:
                mean_val = tsne
            else:
                mean_val += tsne
            count += 1
        mean_val /= count
        return mean_val

    def plot_graphs(self):
        plot_size = [1000, 1000]
        max_image_size = 200
        fig = None
        file_name = "./scatter.dat"
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as fh:
                dat_list = pickle.load(fh)
        for i in range(0, 3):
            tx = dat_list[0 + 4 * i]
            ty = dat_list[1 + 4 * i]
            images = dat_list[2 + 4 * i]
            labels = dat_list[3 + 4 * i]
            fig_no = 231 + i
            fig = self.visualize_tsne_points(tx, ty, labels, fig, fig_no)

        for i in range(0, 3):
            tx = dat_list[0 + 4 * i]
            ty = dat_list[1 + 4 * i]
            images = dat_list[2 + 4 * i]
            labels = dat_list[3 + 4 * i]
            fig_no = 234 + i
            self.visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size,
                                       fig=fig, fig_no=fig_no)