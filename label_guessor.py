import torch

class LabelGuessor(object):
    def __init__(self, model, T):
        self.T = T
        self.guessor = model

    @torch.no_grad()
    def __call__(self, ema, ims):
        org_state = {
            k: v.clone().detach()
            for k, v in self.guessor.state_dict().items()
        }
        is_train = self.guessor.training
        self.guessor.load_state_dict({
            k: v.clone().detach()
            for k, v in ema.state_dict.items()
        })
        self.guessor.train()
        all_probs = []
        for im in ims:
            im = im.cuda()
            logits = self.guessor(im)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
        qb = sum(all_probs)/len(all_probs)
        lbs_tem = torch.pow(qb, 1./self.T)
        lbs = lbs_tem / torch.sum(lbs_tem, dim=1, keepdim=True)
        self.guessor.load_state_dict(org_state)
        if not is_train: self.guessor.eval()
        return lbs.detach()

