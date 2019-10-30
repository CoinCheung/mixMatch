import copy
import torch

class LabelGuessor(object):
    def __init__(self, model, T):
        self.T = T
        self.guessor = copy.deepcopy(model)

    #  def __call__(self, model, ims):
    #      self.guessor.load_state_dict(copy.deepcopy(model.state_dict()))
    #      self.guessor.train()
    #      all_probs = []
    #      with torch.no_grad():
    #          for im in ims:
    #              im = im.cuda()
    #              logits = self.guessor(im)
    #              probs = torch.softmax(logits, dim=1)
    #              all_probs.append(probs)
    #          qb = sum(all_probs)/len(all_probs)
    #          lbs_tem = torch.pow(qb, 1./self.T)
    #          lbs = lbs_tem / torch.sum(lbs_tem, dim=1, keepdim=True)
    #      return lbs.detach()

    def __call__(self, ema, ims):
        self.guessor.load_state_dict(copy.deepcopy(ema.state_dict))
        self.guessor.train()
        all_probs = []
        with torch.no_grad():
            for im in ims:
                im = im.cuda()
                logits = self.guessor(im)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
            qb = sum(all_probs)/len(all_probs)
            lbs_tem = torch.pow(qb, 1./self.T)
            lbs = lbs_tem / torch.sum(lbs_tem, dim=1, keepdim=True)
        return lbs.detach()

