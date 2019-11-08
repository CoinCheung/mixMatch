import torch


class EMA(object):
    def __init__(self, model, alpha, wd, lr):
        self.model = model
        self.alpha = alpha
        self.decay = (1 - wd * lr)
        #  self.state_dict = copy.deepcopy(model.state_dict())
        self.state_dict = {
            k: v.clone().detach()
            for k, v in model.state_dict().items()
        }
        self.buffer_keys, self.param_keys = [], []
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [
            k for k in self.state_dict.keys() if not k in self.param_keys
        ]

    def update_params(self):
        md = self.model.state_dict()
        for name in self.param_keys:
            s, m = self.state_dict[name], md[name]
            #  self.state_dict[name] = self.decay*(self.alpha*s + (1-self.alpha)*m)
            #  md[name] *= self.decay
            md[name] = m * self.decay
            self.state_dict[name] = self.alpha*s + (1-self.alpha)*m
        self.model.load_state_dict(md)

    def update_buffer(self):
        md = self.model.state_dict()
        for name in self.buffer_keys:
            self.state_dict[name] = md[name]

    def save_model(self, pth):
        torch.save(self.state_dict, pth)



if __name__ == '__main__':
    #  print(sd)
    #  print(model.state_dict())
    #  print('=====')
    #  for name, _ in model.named_parameters():
    #      print(name)
    #  print('=====')
    #  for name, _ in model.state_dict().items():
    #      print(name)
    #  print('=====')
    #  print(sd)
    #  model.load_state_dict(sd)
    #  print(model.state_dict())
    #  out = model(inten)
    #  print(sd)
    #  print(model.state_dict())
    import torch.nn as nn
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(3, 8, 3, 1, 1)
            self.bn = nn.BatchNorm2d(8)
            self.act = nn.ReLU()
            self.linear = nn.Linear(8, 5)
        def forward(self, x):
            feat = self.act(self.bn(self.conv(x)))
            feat = torch.mean(feat, dim=(2, 3))
            logits = self.linear(feat)
            return logits

    print('=====')
    import torchvision
    model = Model()
    criteria = torch.nn.CrossEntropyLoss()
    ema = EMA(model, 0.9, 0.02, 0.002)
    optim = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    inten = torch.randn(8, 3, 224, 224)
    lbs = torch.randint(0, 5, (8, ))
    out = model(inten)
    loss = criteria(out, lbs)
    optim.zero_grad()
    loss.backward()
    md = model
    print(md.state_dict()['bn.weight'])
    print('=======')
    print(ema.state_dict['bn.weight'])
    print('=======')
    optim.step()
    print(md.state_dict()['bn.weight'])
    print('=======')
    print(ema.state_dict['bn.weight'])
    print('=======')
    ema.update_params()
    print(md.state_dict()['bn.weight'])
    print('=======')
    print(ema.state_dict['bn.weight'])
    print('=======')
    ema.update_buffer()
    print(md.state_dict()['bn.weight'])
    print('=======')
    print(ema.state_dict['bn.weight'])
    print('=======')
