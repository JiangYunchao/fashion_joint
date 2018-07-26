"""Models for fashion Net."""
import torch
from .basemodel import FashionBase


class FashionNet(FashionBase):
    """Fashion Net.
    Fashion Net has three parts:
    1. features: Feature extractor for all items.
    2. encoder: Learn latent code for each category.
    3. user embdding: Embdding user to latent codes.
    """

    def __init__(self, num, dim, **kwargs):
        """Initialize FashionNet.
        Parameters:
        num: number of users.
        dim: Dimension for user latent code.
        """
        super(FashionNet, self).__init__(num, dim, **kwargs)
        self.num_users = num
        self.dim = dim
        #self.scale_tanh = scale_tanh



class FashionNetDeploy(FashionNet):
    """Fashion Net.
    Fashion Net has three parts:
    1. features: Feature extractor for all items.
    2. encoder: Learn latent code for each category.
    3. user embdding: Embdding user to latent codes.
    """

    def __init__(self, num, dim, **kwargs):
        """Initialize FashionNet.
        Parameters:
        num: number of users.
        dim: Dimension for user latent code.
        """
        super(FashionNetDeploy, self).__init__(num, dim, **kwargs)

    #def binary(self):
     #   """Compute scores with binary latent codes."""
      #  user_codes = self.user_codes.sign()
       # item_codes = [h.sign() for h in self.item_codes]
        # compute scores
        #uscore = self.uscores(user_codes, item_codes)
        #iscore = self.iscores(item_codes)
        #score = (uscore + iscore) / self.dim
        #return score

 #   def forward(self, items, uidx):
        """Forward.
        Return the scores for items.
        """
        # compute latent codes
  #      self.user_codes = self.ulatent_codes(uidx)
   #     self.item_codes = self.ilatent_codes(items)
    #    uscore = self.uscores(self.user_codes, self.item_codes)
     #   iscore = self.iscores(self.item_codes)
      #  score = (uscore + iscore) / self.dim
       # return score
    def forward(self, *input):
        """Forward.

        Return the scores for items.
        """
        
        item_text, item_img, uidx = input
        # compute latent codes
        user_codes = self.user_embdding(uidx)
        item_img_codes = self.ilatent_img_codes(item_img)
        item_text_codes = self.ilatent_text_codes(item_text)
        #ubianry = self.hash(user_codes)
        #ibinary = [self.hash(h) for h in item_codes]
        uscore_img = self.uscores(user_codes, item_img_codes)
        iscore_img = self.iscores(item_img_codes)
        uscore_text = self.uscores(user_codes, item_text_codes)
        iscore_text = self.iscores(item_text_codes)
        score = uscore_img + uscore_text + iscore_img + iscore_text

        binary = torch.zeros_like(score)#self.scores(ubianry, ibinary).view(-1, 1)
        return score, binary
