import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models
import numpy as np
import os
import subprocess
import math
import argparse
import random
import cv2

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 1)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 5)
parser.add_argument("-e","--episode",type = int, default= 800000)
parser.add_argument("-start","--start_episode",type = int, default= 0)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-d","--display_query_num",type=int,default=5)
parser.add_argument("-ex","--exclude_class",type=int,default=6)
parser.add_argument("-modelf","--feature_encoder_model",type=str,default='')
parser.add_argument("-modelr","--relation_network_model",type=str,default='')
parser.add_argument("-lo","--loadImagenet",type=bool,default=False)
parser.add_argument("-fi","--finetune",type=bool,default=True)
parser.add_argument("-rf","--TrainResultPath",type=str,default='result_newvgg_1shot')
parser.add_argument("-rff","--ResultSaveFreq",type=int,default=10000)
parser.add_argument("-msp","--ModelSavePath",type=str,default='models_newvgg_1shot')
parser.add_argument("-msf","--ModelSaveFreq",type=int,default=10000)


args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
DISPLAY_QUERY = args.display_query_num
EXCLUDE_CLASS = args.exclude_class
FEATURE_MODEL = args.feature_encoder_model
RELATION_MODEL = args.relation_network_model

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        features = list(models.vgg16_bn(pretrained=args.loadImagenet).features)
        self.layer1 = nn.Sequential(
                        nn.Conv2d(4,64,kernel_size=3,padding=1)
                        )
        self.features = nn.ModuleList(features)[1:]#.eval()
        # print (nn.Sequential(*list(models.vgg16_bn(pretrained=True).children())[0]))
        # self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        x = self.layer1(x)
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 11, 21, 31, 41}:
                results.append(x)

        return x, results

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1024,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
                        nn.Conv2d(1024,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(512,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 14 x 14
        self.double_conv2 = nn.Sequential(
                        nn.Conv2d(1024,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(256,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 28 x 28
        self.double_conv3 = nn.Sequential(
                        nn.Conv2d(512,128,kernel_size=3,padding=1),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(128,128,kernel_size=3,padding=1),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 56 x 56
        self.double_conv4 = nn.Sequential(
                        nn.Conv2d(256,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU()
                        ) # 112 x 112
        self.double_conv5 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(64,1,kernel_size=1,padding=0),
                        ) # 256 x 256

    def forward(self,x,concat_features):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.upsample(out) #block 1
        out = torch.cat((out, concat_features[-1]), dim=1)
        out = self.double_conv1(out)
        out = self.upsample(out) #block 2
        out = torch.cat((out, concat_features[-2]), dim=1)
        out = self.double_conv2(out)
        out = self.upsample(out) #block 3
        out = torch.cat((out, concat_features[-3]), dim=1)
        out = self.double_conv3(out)
        out = self.upsample(out) #block 4
        out = torch.cat((out, concat_features[-4]), dim=1)
        out = self.double_conv4(out)
        out = self.upsample(out) #block 5
        out = torch.cat((out, concat_features[-5]), dim=1)
        out = self.double_conv5(out)

        out = F.sigmoid(out)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())



def get_oneshot_batch():  #shuffle in query_images not done
    # classes.remove(EXCLUDE_CLASS)
    classes_name = os.listdir('./fewshot/support/')
    classes = list(range(0,len(classes_name)))

    chosen_classes = random.sample(classes, CLASS_NUM)
    support_images = np.zeros((CLASS_NUM*SAMPLE_NUM_PER_CLASS,3,224,224), dtype=np.float32)
    support_labels = np.zeros((CLASS_NUM*SAMPLE_NUM_PER_CLASS,CLASS_NUM,224,224), dtype=np.float32)
    query_images = np.zeros((CLASS_NUM*BATCH_NUM_PER_CLASS,3,224,224), dtype=np.float32)
    query_labels = np.zeros((CLASS_NUM*BATCH_NUM_PER_CLASS,CLASS_NUM,224,224), dtype=np.float32)
    zeros = np.zeros((CLASS_NUM*BATCH_NUM_PER_CLASS,1,224,224), dtype=np.float32)
    class_cnt = 0
    for i in chosen_classes:
        # print ('class %s is chosen' % i)
        imgnames = os.listdir('./fewshot/support/%s/label' % classes_name[i])
        indexs = list(range(0,len(imgnames)))
        chosen_index = random.sample(indexs, SAMPLE_NUM_PER_CLASS + BATCH_NUM_PER_CLASS)
        j = 0
        for k in chosen_index:
            # process image
            image = cv2.imread('./fewshot/support/%s/image/%s' % (classes_name[i], imgnames[k].replace('.png', '.jpg')))
            if image is None:
                print ('./fewshot/support/%s/image/%s' % (classes_name[i], imgnames[k].replace('.png', '.jpg')))
                stop
            image = image[:,:,::-1] # bgr to rgb
            image = image / 255.0
            image = np.transpose(image, (2,0,1))
            # labels
            label = cv2.imread('./fewshot/support/%s/label/%s' % (classes_name[i], imgnames[k]))[:,:,0]
            if j < SAMPLE_NUM_PER_CLASS:
                support_images[j] = image
                support_labels[j][0] = label
            else:
                query_images[j-SAMPLE_NUM_PER_CLASS] = image
                query_labels[j-SAMPLE_NUM_PER_CLASS][class_cnt] = label
            j += 1

        class_cnt += 1
    support_images_tensor = torch.from_numpy(support_images)
    support_labels_tensor = torch.from_numpy(support_labels)
    support_images_tensor = torch.cat((support_images_tensor,support_labels_tensor), dim=1)

    zeros_tensor = torch.from_numpy(zeros)
    query_images_tensor = torch.from_numpy(query_images)
    query_images_tensor = torch.cat((query_images_tensor,zeros_tensor), dim=1)
    query_labels_tensor = torch.from_numpy(query_labels)

    return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor, chosen_classes


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors

    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0],
                      [0,0,128], [128,0,128], [0,128,128], [128,128,128],
                      [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                      [64,0,128], [192,0,128], [64,128,128], [192,128,128],
                      [0, 64,0], [128, 64, 0], [0,192,0], [128,192,0],
                      [0,64,128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_segmap(label_mask, plot=False):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_pascal_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r #/ 255.0
    rgb[:, :, 1] = g #/ 255.0
    rgb[:, :, 2] = b #/ 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb




def main():

    # Step 1: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork()

    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    # fine-tuning
    if (args.finetune):
        if os.path.exists(FEATURE_MODEL):
            feature_encoder.load_state_dict(torch.load(FEATURE_MODEL))
            print("load feature encoder success")
        else:
            print('Can not load feature encoder: %s' % FEATURE_MODEL)
            print('starting from scratch')
        if os.path.exists(RELATION_MODEL):
            relation_network.load_state_dict(torch.load(RELATION_MODEL))
            print("load relation network success")
        else:
            print('Can not load relation network: %s' % RELATION_MODEL)
            print('starting from scratch')
    


    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=EPISODE//10,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=EPISODE//10,gamma=0.5)


    print("Training...")

    last_accuracy = 0.0

    for episode in range(args.start_episode, EPISODE):
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        samples, sample_labels, batches, batch_labels, chosen_classes = get_oneshot_batch()


        # calculate features
        sample_features, _ = feature_encoder(Variable(samples).cuda(GPU))
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,512,7,7)
        sample_features = torch.sum(sample_features,1).squeeze(1) # 1*512*7*7
        batch_features, ft_list = feature_encoder(Variable(batches).cuda(GPU))

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)

        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,1024,7,7)
        output = relation_network(relation_pairs,ft_list).view(-1,CLASS_NUM,224,224)

        mse = nn.MSELoss().cuda(GPU)
        loss = mse(output,Variable(batch_labels).cuda(GPU))


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1)%10 == 0:
                print("episode:",episode+1,"loss",loss.cpu().data.numpy())

        if not os.path.exists(args.TrainResultPath):
            os.makedirs(args.TrainResultPath)
        if not os.path.exists(args.ModelSavePath):
            os.makedirs(args.ModelSavePath)

        # training result visualization
        if (episode+1)%args.ResultSaveFreq == 0:
            support_output = np.zeros((224*2, 224*SAMPLE_NUM_PER_CLASS, 3), dtype=np.uint8)
            query_output = np.zeros((224*3, 224*DISPLAY_QUERY, 3), dtype=np.uint8)
            chosen_query = random.sample(list(range(0,BATCH_NUM_PER_CLASS)), DISPLAY_QUERY)

            for i in range(CLASS_NUM):
                for j in range(SAMPLE_NUM_PER_CLASS):
                    supp_img = (np.transpose(samples.numpy()[j],(1,2,0))*255).astype(np.uint8)[:,:,:3][:,:,::-1]
                    support_output[0:224,j*224:(j+1)*224,:] = supp_img
                    supp_label = sample_labels.numpy()[j][0]
                    supp_label[supp_label!=0] = chosen_classes[i]
                    supp_label = decode_segmap(supp_label)
                    support_output[224:224*2, j*224:(j+1)*224,:] = supp_label

                for cnt, x in enumerate(chosen_query):
                    query_img = (np.transpose(batches.numpy()[x],(1,2,0))*255).astype(np.uint8)[:,:,:3][:,:,::-1]
                    query_output[0:224,cnt*224:(cnt+1)*224,:] = query_img
                    query_label = batch_labels.numpy()[x][0] #only apply to one-way setting
                    query_label[query_label!=0] = chosen_classes[i]
                    query_label = decode_segmap(query_label)
                    query_output[224:224*2, cnt*224:(cnt+1)*224,:] = query_label

                    query_pred = output.detach().cpu().numpy()[x][0]
                    query_pred = (query_pred*255).astype(np.uint8)
                    result = np.zeros((224,224,3), dtype=np.uint8)
                    result[:,:,0] = query_pred
                    result[:,:,1] = query_pred
                    result[:,:,2] = query_pred
                    query_output[224*2:224*3, cnt*224:(cnt+1)*224,:] = result
            extra = query_output.copy()
            for i in range(CLASS_NUM):
                for cnt, x in enumerate(chosen_query):
                    extra_label = batch_labels.numpy()[x][0]
                    extra_label[extra_label!=0] = 255
                    result1 = np.zeros((224,224,3), dtype=np.uint8)
                    result1[:,:,0] = extra_label
                    result1[:,:,1] = extra_label
                    result1[:,:,2] = extra_label
                    extra[224*2:224*3, cnt*224:(cnt+1)*224,:] = result1
            cv2.imwrite('%s/%s_query.png' % (args.TrainResultPath, episode), query_output)
            cv2.imwrite('%s/%s_show.png' % (args.TrainResultPath,episode), extra)
            cv2.imwrite('%s/%s_support.png' % (args.TrainResultPath, episode), support_output)

        #save models
        if (episode+1) % args.ModelSaveFreq == 0:
            torch.save(feature_encoder.state_dict(),str("./%s/feature_encoder_" % args.ModelSavePath + str(episode) + '_' + str(CLASS_NUM) +"_way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
            torch.save(relation_network.state_dict(),str("./%s/relation_network_" % args.ModelSavePath + str(episode) + '_' + str(CLASS_NUM) +"_way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
            print("save networks for episode:",episode)


if __name__ == '__main__':
    main()
