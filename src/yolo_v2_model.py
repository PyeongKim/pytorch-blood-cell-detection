import torch.nn as nn
import torch

class Yolo_v2(nn.Module):
  def __init__(self, num_classes=3, num_boxes=5):
    super(Yolo_v2, self).__init__()
    self.num_classes = num_classes
    self.num_boxes = num_boxes

    # First Stage
    self.layer_1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                 nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
    self.layer_2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
    self.layer_3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
    self.layer_6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False),nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
    self.layer_9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False),nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False),nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.1, inplace=True))
    
    # Second Stage (a)
    self.maxpooling2d = nn.MaxPool2d(2, 2)
    self.layer_14 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_15 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False),nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_16 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_17 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False),nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_18 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_19 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.layer_20 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(0.1, inplace=True))
    # Second Stage (b)
    self.layer_21 = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False),nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.1, inplace=True))
    self.block_size = 2
    self.space_to_depth_x2 = nn.Unfold(kernel_size=(self.block_size, self.block_size), stride=2)

    # Final Stage
    self.layer_22 = nn.Sequential(nn.Conv2d(256 + 1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                  nn.LeakyReLU(0.1, inplace=True))
    self.layer_23 = nn.Sequential(nn.Conv2d(1024, self.num_boxes*(4+1+self.num_classes), 1, 1, 0, bias=False), 
                                  nn.LeakyReLU(0.1, inplace=True))   
    
  def forward(self, input_image):
    # First Stage
    output = self.layer_1(input_image)
    output = self.layer_2(output)
    output = self.layer_3(output)
    output = self.layer_4(output)
    output = self.layer_5(output)
    output = self.layer_6(output)
    output = self.layer_7(output)
    output = self.layer_8(output)
    output = self.layer_9(output)
    output = self.layer_10(output)
    output = self.layer_11(output)
    output = self.layer_12(output)
    output = self.layer_13(output)
    print("First Stage: ",output.size())

    skip_connection = output
    print("skip: ", skip_connection.size())

    # Second Stage (a)
    output = self.maxpooling2d(output)
    output = self.layer_14(output)
    output = self.layer_15(output)
    output = self.layer_16(output)
    output = self.layer_17(output)
    output = self.layer_18(output)
    output = self.layer_19(output)
    output = self.layer_20(output)

    # Second Stage (b)
    skip_connection = self.layer_21(skip_connection)
    n, c, h, w = skip_connection.size()
    skip_connection = self.space_to_depth_x2(skip_connection)
    skip_connection = skip_connection.view(n, c * self.block_size**2, h // self.block_size, w // self.block_size)
    print("skip_: ",skip_connection.size())
    output = torch.cat((output, skip_connection), 1)
    
    # Final Stage
    output = self.layer_22(output)
    output = self.layer_23(output)
    print(output.size())

    return output
