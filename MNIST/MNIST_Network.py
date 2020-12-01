# conv1-conv11-pool-conv2-conv21

class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()
        self.conv1    = nn.Conv2d(1,32,kernel_size=5,dilation=1, stride=1, padding=2,bias=True)
        self.conv11   = nn.Conv2d(32,32,kernel_size=5,dilation=1, stride=1, padding=2,bias=True) 
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32,64,kernel_size=5,dilation=1, stride=1, padding=2,bias=True)
        self.conv21 = nn.Conv2d(64,64,kernel_size=5,dilation=1, stride=1, padding=2,bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1   = nn.Linear(64*7*7,512,bias=True )
        self.fc2   = nn.Linear(512, 10)
    def forward(self, input):
        out = F.relu((self.conv1(input)))
        out = F.relu((self.conv11(out)))
        out = self.pool1(out)
        
        out = F.relu((self.conv2(out)))
        out = F.relu((self.conv21(out)))
        out = self.pool2(out)
        
        # fc-1
        B,C,H,W = out.size()
        out = out.view(B,-1) 
        out =(F.relu((self.fc1(out))))
        # Logits
        out = self.fc2(out)
        return out 
    
