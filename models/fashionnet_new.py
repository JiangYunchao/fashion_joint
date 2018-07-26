"""Models for fashion Net."""
import torch
from .basemodel import FashionBase, LearnableScale


class FashionNet(FashionBase):
    """Fashion Net.

    Fashion Net has three parts:
    1. features: Feature extractor for all items.
    2. encoder: Learn latent code for each category.
    3. user embdding: Embdding user to latent codes.
    """

    def __init__(self, num, dim, param):
        """Initialize FashionNet.

        Parameters:
        num: number of users.
        dim: Dimension for user latent code.

        """
        super(FashionNet, self).__init__(num, dim, param)
        self.beta = LearnableScale(1)

    def debug(self):
        self.user_embdding.debug()
        self.beta.debug()

    def uscores(self, ulatent, ilatents):
        """Compute u-term in scores.

        If zero_uscores is True, return 0

        Parameters
        ----------
        ulatent: user latent code
        ilatents: item latent codes

        """
        scores = torch.zeros_like(ilatents[0])
        if self.zero_uscores:
            return scores.sum(dim=1)
        for latent in ilatents:
            scores += ulatent * latent
        scores = self.beta(scores.sum(dim=1))
        return scores


class FashionNetDeploy(FashionNet):
    """Fashion Net.

    Fashion Net has three parts:
    1. features: Feature extractor for all items.
    2. encoder: Learn latent code for each category.
    3. user embdding: Embdding user to latent codes.
    """

    def __init__(self, num, dim, param):
        """Initialize FashionNet.

        Parameters:
        num: number of users.
        dim: Dimension for user latent code.

        """
        super(FashionNetDeploy, self).__init__(num, dim, param)

    def forward(self, *input):
        """Forward.

        Return the scores for items.
        """
        items, uidx = input
        # compute latent codes
        user_codes = self.user_embdding(uidx)
        item_codes = self.ilatent_codes(items)
        ubianry = self.hash(user_codes)
        ibinary = [self.hash(h) for h in item_codes]
        score = self.scores(user_codes, item_codes).view(-1, 1)
        binary = self.scores(ubianry, ibinary).view(-1, 1)
        return score, binary
