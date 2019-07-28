import copy
import torch


class EMA(object):
    def __init__(self, model, alpha, wd, lr):
        self.model = model
        self.alpha = alpha
        self.decay = (1 - wd * lr)
        self.state_dict = copy.deepcopy(model.state_dict())
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
            self.state_dict[name] = self.alpha*s + (1-self.alpha)*m
            md[name] = self.decay * m
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

    print('=====')
    model = torch.nn.BatchNorm1d(5)
    ema = EMA(model, 0.9, 0.02, 0.002)
    inten = torch.randn(10, 5)
    out = model(inten)
    ema.update_params()
    print(model.state_dict())
    ema.update_buffer()
    print(model.state_dict())
