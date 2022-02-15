criterion_scratch = nn.NLLLoss()

def get_optimizer_scratch(model):
    return optim.Adam(model.parameters()) 

# maxpool only

import torch.nn as nn

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # sees 224x224x3 tesnsor
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        # sees 112x112x16 tensor
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        # sees 56x56x32 tensor
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        # sees 28x28x64 tensor
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        # outputs 14x14x64 tensor
        # expects flattened tensor with 12544 features
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(12544, 500)),
            # ('fc1_bn', nn.BatchNorm1d(500)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(500, 250)),
            ('relu2', nn.ReLU()),
            # ('fc2_bn', nn.BatchNorm1d(250)),
            ('dropout2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(250, 100)),
            # ('fc3_bn', nn.BatchNorm1d(100)),
            ('relu3', nn.ReLU()),
            ('dropout3', nn.Dropout(p=0.2)),
            ('fc_final', nn.Linear(100, 50)),
            # ('fc_final_bn', nn.BatchNorm1d(50)),
            ('log_output', nn.LogSoftmax(dim=1))
        ]))
    
    def forward(self, x):


        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 14 * 14 * 64)

        x = self.classifier(x)
        
        self.log_ps = x
        return x

#-#-# Do NOT modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()