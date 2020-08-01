
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dim, 512)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(512, 64)
        self.affine3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)

class PolicyCNN(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(PolicyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.fc1 = nn.Linear(250,64)
        self.fc2 = nn.Linear(64, action_dim)

    # action output between -2 and 2
    #@torchsnooper.snoop()
    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        result = F.softmax(x)
        return result

class socialMask(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(socialMask, self).__init__()
        self.affine1 = nn.Linear(state_dim, 512)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(512, 64)
        self.affine3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = F.relu(self.affine2(x))
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)

class CNN_preprocess(nn.Module):
    def __init__(self,width,height,channel):
        super(CNN_preprocess,self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=(1,1))
        self.Conv2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=5,stride=5)
        self.Conv3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=3)


    def forward(self,x):
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Conv3(x)
        x = F.relu(x)
        return torch.flatten(x)

    def get_state_dim(self):
        return 64

class Actor(nn.Module):
    def __init__(self,action_dim,state_dim):
        super(Actor,self).__init__()
        self.Linear1 = nn.Linear(state_dim,128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128,128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128,action_dim)

    def forward(self,x):
        x = self.Linear1(x)
        # x = self.Dropout1(x)
        x = F.relu(x)
        # x = self.Linear2(x)
        # x = self.Dropout2(x)
        # x = F.relu(x)
        x = self.Linear3(x)
        return F.softmax(x)

class Critic(nn.Module):
    def __init__(self,state_dim):
        super(Critic,self).__init__()
        self.Linear1 = nn.Linear(state_dim, 128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128, 128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128,1)

    def forward(self,x):
        x = self.Linear1(x)
        # x = self.Dropout1(x)
        x = F.relu(x)
        # x = self.Linear2(x)
        # x = self.Dropout2(x)
        # x = F.relu(x)
        x = self.Linear3(x)
        return x

class Centralised_Critic(nn.Module):
    def __init__(self,state_dim):
        super(Centralised_Critic,self).__init__()
        self.Linear1 = nn.Linear(state_dim*2,128)
        self.Linear2 = nn.Linear(128,1)

    def forward(self,x):
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x

class ActorRNN(nn.Module):
    def __init__(self,state_dim,action_dim,CNN=True):
        super(ActorRNN, self).__init__()
        self.CNN = CNN
        if CNN:
            # self.Conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=(1,1))
            self.Conv2 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,stride=1,padding=(2,2))
            self.Pool2 = nn.MaxPool2d(kernel_size=5,stride=5)
            self.Conv3 = nn.Conv2d(in_channels=64,out_channels=16,kernel_size=3,stride=1,padding=(1,1))
            self.Pool3 = nn.MaxPool2d(kernel_size=3,stride=3)
            self.rnn = nn.GRU(
                input_size=16,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128,action_dim)
        else:
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128,action_dim)

    def forward(self,x):
        if self.CNN:
            # x = self.Conv1(x)
            # x = F.relu(x)
            x = self.Conv2(x)
            x = F.relu(x)
            x = self.Pool2(x)
            x = self.Conv3(x)
            x = F.relu(x)
            x = self.Pool3(x)
            x = torch.flatten(x,start_dim=1,end_dim=-1).unsqueeze(0)
        x, h_n = self.rnn(x,None)
        x = F.relu(x)
        x = self.out(x)[:,-1,:]
        return F.softmax(x)

class CriticRNN(nn.Module):
    def __init__(self,state_dim,action_dim,CNN=True):
        super(CriticRNN, self).__init__()
        self.CNN = CNN
        if CNN:
            # self.Conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=(1,1))
            self.Conv2 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=5,stride=1,padding=(2,2))
            self.Pool2 = nn.MaxPool2d(kernel_size=5,stride=5)
            self.Conv3 = nn.Conv2d(in_channels=64,out_channels=16,kernel_size=3,stride=1,padding=(1,1))
            self.Pool3 = nn.MaxPool2d(kernel_size=3,stride=3)
            self.rnn = nn.GRU(
                input_size=16,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128,1)
        else:
            self.rnn = nn.GRU(
                input_size=state_dim,
                hidden_size=128,
                num_layers=1,
            )
            self.out = nn.Linear(128,1)

    def forward(self,x):
        if self.CNN:
            # x = self.Conv1(x)
            # x = F.relu(x)
            x = self.Conv2(x)
            x = F.relu(x)
            x = self.Pool2(x)
            x = self.Conv3(x)
            x = F.relu(x)
            x = self.Pool3(x)
            x = torch.flatten(x,start_dim=1,end_dim=-1).unsqueeze(0)
        x, h_n = self.rnn(x,None)
        x = F.relu(x)
        x = self.out(x)[:,-1,:]
        return x



if __name__ == "__main__":
    model_name = "pg_social"
    file_name  = "train_para/"+model_name
    agentParam = {"ifload":True,"filename": file_name,"id":"0"}
    net = torch.load(agentParam["filename"]+"pg"+agentParam["id"]+".pth",map_location = torch.device('cuda'))
    optimizer = optim.Adam(net.parameters(), lr=0.01)
