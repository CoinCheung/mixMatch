import torch

class LabelGuessor(object):
    def __init__(self, T):
        self.T = T

    def __call__(self, model, ims):
        model.train()
        all_probs = []
        with torch.no_grad():
            for im in ims:
                im = im.cuda()
                logits = model(im)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
            qb = sum(all_probs)/len(all_probs)
            lbs_tem = torch.pow(qb, 1./self.T)
            lbs = lbs_tem / torch.sum(lbs_tem, dim=1, keepdim=True)
        return lbs.detach()

