import torch



class Encoder(torch.nn.Module):
    def __init__(self,input_dim=2, hidden_dim=10, output_dim=2, phase=True):
        self.e = 1.0e-10
        self.phase = phase
        super(Encoder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
            
    def forward(self, x):
        y = self.model(x)
        if self.phase:
            r = torch.norm(y[:,:2]+self.e,dim=1)
            r = r.tile((2,1)).T
            z1 = y[:,:2]/r
            z = torch.cat((z1,y[:,2:]),dim=1)
            return z
        else:
            return y

class Decoder(torch.nn.Module):
    def __init__(self,input_dim=2, hidden_dim=10, output_dim=2):
        super(Decoder, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
            #torch.nn.Linear(hidden_dim, 3),
            #torch.nn.BatchNorm1d(3),
            #torch.nn.ReLU(),
            #torch.nn.Linear(3, output_dim)
        )

    def forward(self, x):
        y = self.model(x)
        return y
    
class LatentSteper(torch.nn.Module):
    def __init__(self,zd=1):
        super(LatentSteper, self).__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.01))
        self.lam = torch.nn.Parameter(torch.tensor([0.99]*zd))

    def forward(self, input):
        z0 = input[:,[0]] * torch.cos(self.theta) - input[:,[1]] * torch.sin(self.theta)
        z1 = input[:,[0]] * torch.sin(self.theta) + input[:,[1]] * torch.cos(self.theta)
        z2 = input[:,2:] * self.lam
        z = torch.cat((z0,z1,z2),dim=1)

        return z