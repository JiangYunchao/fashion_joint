"""Models for fashion Net."""
import torch

from torch import nn
import config as cfg
NUM_ENCODER = cfg.NumCate
import torch.nn.functional as F

class ItemImgFeature(nn.Module):
    """AlexNet for feature extractor."""

    def __init__(self):
        """Feature Extractor.

        Extract the feature for item.
        """
        super(ItemImgFeature, self).__init__()
        self.dim = 4096
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*8*8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
)

    def forward(self, x):
        """Forward."""
        x = self.features(x)
        x = x.view(x.size(0), 256*8*8)
        x = self.classifier(x)
        return x.view(-1, self.dim)

class ItemTextFeature(nn.Module):
    """AlexNet for feature extractor."""

    def __init__(self):
        """Feature Extractor.

        Extract the feature for item.
        """
        super(ItemTextFeature, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(2400, 1024),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        """Forward."""
        x = self.features(x)
        return x


class ItemImgEncoder(nn.Module):
    """Module for latent code.

    Encoder for item's features.
    """

    def __init__(self, dim):
        """Initialize an encoder.

        Parameter
        ---------
        dim: Dimension for latent space

        """
        super(ItemImgEncoder, self).__init__()
        self.register_buffer('scale', torch.ones(1))
        self.encoder = nn.Sequential(
            nn.Linear(4096, dim),
            nn.ReLU()
        )
        self.active = nn.Tanh()

    def set_scale(self, value):
        """Set scale tanh."""
        self.scale.fill_(value)

    def forward(self, x):
        """Forward a feature from ItemFeature."""
        x = self.encoder(x)
        h = torch.mul(x, torch.autograd.Variable(self.scale))
        return self.active(h)

class ItemTextEncoder(nn.Module):
    """Module for latent code.

    Encoder for item's features.
    """

    def __init__(self, dim):
        """Initialize an encoder.

        Parameter
        ---------
        dim: Dimension for latent space

        """
        super(ItemTextEncoder, self).__init__()
        self.register_buffer('scale', torch.ones(1))
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, dim),
            nn.ReLU()
        )
        self.active = nn.Tanh()

    def set_scale(self, value):
        """Set scale tanh."""
        self.scale.fill_(value)

    def forward(self, x):
        """Forward a feature from ItemFeature."""
        x = self.encoder(x)
        h = torch.mul(x.float(), torch.autograd.Variable(self.scale))
        return self.active(h)

# class ItemEncoder2C(nn.Module):
#     """Module for latent code with 2 channels."""
#
#     def __init__(self, dim):
#         """Initialize an encoder.
#
#         Parameters
#         ----------
#         dim: Dimension for latent space
#
#         """
#         super(ItemEncoder2C, self).__init__()
#         self.uencoder = ItemEncoder(dim)
#         self.iencoder = ItemEncoder(dim)
#
#     def forward(self, x):
#         """Forward a feature from ItemFeature."""
#         ucode = self.uencoder(x)
#         icode = self.iencoder(x)
#         return (ucode, icode)
#
#     def init_weights(self, state_dict):
#         """Initialize weights for two encoders."""
#         for model in self.children():
#             model.init_weights(state_dict)


class UserEncoder(nn.Module):
    """User embdding layer."""

    def __init__(self, num_users, dim):
        """User embdding.

        Parameters:
        ----------
        num_users: number of users.
        dim: Dimension for user latent code.
        single: if use single layer to learn user's preference.
        linear: if user Linear layer to learn user's preference.

        """
        super(UserEncoder, self).__init__()
        self.register_buffer('scale', torch.ones(1))
        self.embdding = nn.Linear(num_users, dim, bias=False)
        self.active = nn.Tanh()

    def set_scale(self, value):
        """Set scale tanh."""
        self.scale.fill_(value)

    def init_weights(self, state_dict=None):
        """Initialize weights for user encoder."""
        for param in self.parameters():
            param.data.normal_(0, 0.01)
            # param.data.add_((param.data < -1).type_as(param.data) * 2)

    def forward(self, input):
        """Get user's latent codes given index."""
        x = self.embdding(input)
        h = torch.mul(x, torch.autograd.Variable(self.scale))
        return self.active(h)

# class VSE(nn.Module):
#     """projecting img vector to text vector space"""
#     def __init__(self, dim):
#         super(VSE, self).__init__()
#         self.register_buffer('scale', torch.ones(1))
#         self.embedding = nn.Sequential(
#                         nn.Linear(dim, dim),
#                         nn.ReLU()
#         )
#     def set_scale(self, value):
#         """Set scale tanh."""
#         self.scale.fill_(value)
#     def forward(self, input):
#         x = self.embedding(input)
#         return x

class PotMul(nn.Module):
    """Pointwise multiplication."""

    def __init__(self, dim, mu=1.0, std=0.01):
        """Weights for this layer that is drawn from N(mu, std)."""
        super(PotMul, self).__init__()
        self.mu = mu
        self.std = std
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.init_weights()

    def init_weights(self, state_dict=None):
        """Initialize weights."""
        self.weight.data.normal_(self.mu, self.std)

    def forward(self, input):
        """Forward."""
        return torch.mul(input, self.weight)

    def __repr__(self):
        """Format string for module PotMul."""
        return self.__class__.__name__ + '(dim=' + str(self.dim) + ')'


class FashionBase(nn.Module):
    """Base class for fashion net.

    Methods:
    --------
    accuracy(): return the current accuracy (call after forward()).
    binary(): return the accuracy with binary latent codes.
    loss(): return loss.

    """

    def __init__(self, num_users, dim, single=False):
        """Contols a base instances for FashionNet.

        Parameters:
        num_users: number of users.
        dim: Dimension for user latent code.

        """
        super(FashionBase, self).__init__()
        self.user_embdding = UserEncoder(num_users, dim)
        self.img_features = ItemImgFeature()
        self.text_features = ItemTextFeature()
        #self.embedding = VSE(dim)
        self.single = single
        if single:
            self.img_encoder = ItemImgEncoder(dim)
            self.text_encoder = ItemTextEncoder(dim)
        else:
            self.img_encoder = nn.ModuleList(
                [ItemImgEncoder(dim) for n in range(NUM_ENCODER)])
            self.text_encoder = nn.ModuleList(
                [ItemTextEncoder(dim) for n in range(NUM_ENCODER)])
        self.dim = dim
        self.ratio = 10.0 / self.dim
        self.zero_uscores = False
        self.zero_iscores = False

    def set_scale(self, value):
        """Set scale tahn."""
        self.user_embdding.set_scale(value)
        if self.single:
            self.img_encoder.set_scale(value)
        else:
            for encoder in self.img_encoder:
                encoder.set_scale(value)

    def num_gropus(self):
        """Size of sub-modules."""
        return len(self._modules)

    def name(self):
        """Name of network."""
        return self.__class__.__name__

    def set_zero_uscores(self, flag=True):
        """Set uscores to zero."""
        self.zero_uscores = flag

    def set_zero_iscores(self, flag=True):
        """Set iscores to zero."""
        self.zero_iscores = flag

    def active_all_param(self):
        """Active all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_all_param(self):
        """Active all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_user_param(self):
        """Freeze user's latent codes."""
        self.active_all_param()
        for param in self.user_embdding.parameters():
            param.requires_grad = False

    def freeze_item_param(self):
        """Freeze item's latent codes."""
        self.freeze_all_param()
        for param in self.user_embdding.parameters():
            param.requires_grad = True

    def ilatent_img_codes(self, items):
        """Compute lantent codes for items."""
        ilatent_codes = []
        if self.single:
            for x in items:
                x = self.img_features(x)
                h = self.img_encoder(x)
                ilatent_codes.append(h.view(-1, self.dim))
            return ilatent_codes
        # top items
        for x in items[0:-2]:
            x = self.img_features(x)
            h = self.img_encoder[0](x)
            ilatent_codes.append(h.view(-1, self.dim))
        # others
        for n, x in enumerate(items[-2:]):
            x = self.img_features(x)
            h = self.img_encoder[n + 1](x)
            ilatent_codes.append(h.view(-1, self.dim))
        return ilatent_codes

    def ilatent_text_codes(self, items):
        """Compute lantent codes for items."""
        ilatent_codes = []
        if self.single:
            for x in items:
                x = self.text_features(x)
                h = self.text_encoder(x)
                ilatent_codes.append(h.view(-1, self.dim))
            return ilatent_codes
        # top items
        for x in items[0:-2]:
            x = self.text_features(x)
            h = self.text_encoder[0](x)
            ilatent_codes.append(h.view(-1, self.dim))
        # others
        for n, x in enumerate(items[-2:]):
            x = self.text_features(x)
            h = self.text_encoder[n + 1](x)
            ilatent_codes.append(h.view(-1, self.dim))
        return ilatent_codes

    def img_embedding(self, img):
        img_emb = []
        for x in img:
            x = self.embedding(x)
            img_emb.append(x.view(-1, self.dim))
        return img_emb

    def init_weights(self, state_dict):
        """Initialize net weights with pretrained model.

        Each sub-module should has its own same methods.
        """
        for model in self.children():
            if isinstance(model, nn.ModuleList):
                for m in model:
                    m.init_weights(state_dict)
            else:
                model.init_weights(state_dict)

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
        count = 0
        for latent in ilatents:
            scores += ulatent * latent
            count += 1
        return scores.sum(dim=1) / count

    def iscores(self, ilatents):
        """Compute i-term in scores.

        If zero_iscores is True, return 0

        Parameters
        ----------
        ilatents: item latent codes

        count = 0
        size = len(ilatents)
        #print(ilatents)
        """
        scores = torch.zeros_like(ilatents[0])
        if self.zero_iscores:
            return scores.sum(dim=1)
        size = len(ilatents)
        count = 0
        #print(size)
        for n in range(size):
            for m in range(n + 1, size):
                scores += ilatents[n] * ilatents[m]
                count += 1
        return scores.sum(dim=1) / count

    def forward(self, posi_text, nega_text, posi_img, nega_img, uidx):
        """Forward.

        Return the comparative value between positive and negative tuples.
        """
        # compute latent codes
        user_codes = self.user_embdding(uidx)
        #user_codes_binary = user_codes.detach().sign()
        item_codes_img_posi = self.ilatent_img_codes(posi_img)
        item_codes_img_nega = self.ilatent_img_codes(nega_img)
        item_codes_text_posi = self.ilatent_text_codes(posi_text)
        item_codes_text_nega = self.ilatent_text_codes(nega_text)
        #img_vse = self.img_embedding(item_codes_img_posi)
        #item_codes_nega_binary = [h.detach().sign() for h in item_codes_nega]
        #item_codes_posi_binary = [h.detach().sign() for h in item_codes_posi]
        uscore_img_posi = self.uscores(user_codes, item_codes_img_posi)
        iscore_img_posi = self.iscores(item_codes_img_posi)
        uscore_img_nega = self.uscores(user_codes, item_codes_img_nega)
        iscore_img_nega = self.iscores(item_codes_img_nega)
        uscore_text_posi = self.uscores(user_codes, item_codes_text_posi)
        iscore_text_posi = self.iscores(item_codes_text_posi)
        uscore_text_nega = self.uscores(user_codes, item_codes_text_nega)
        iscore_text_nega = self.iscores(item_codes_text_nega)
        #print("item_codes_posi:{}\nitem_codes_nega:{}".format(item_codes_posi, item_codes_nega))
        #uscore_posi_binary = self.uscores(user_codes_binary,
        #                                  item_codes_posi_binary)
        #iscore_posi_binary = self.iscores(item_codes_posi_binary)
        #uscore_nega_binary = self.uscores(user_codes_binary,
        #                                  item_codes_nega_binary)
        #iscore_nega_binary = self.iscores(item_codes_nega_binary)
        score_posi = uscore_img_posi + iscore_img_posi + uscore_text_posi + iscore_text_posi
        score_nega = uscore_img_nega + iscore_img_nega + uscore_text_nega + iscore_text_nega
        #score_posi_binary = uscore_posi_binary + iscore_posi_binary
        #score_nega_binary = uscore_nega_binary + iscore_nega_binary
        output = self.ratio * (score_posi - score_nega)
        output_binary = self.ratio * (score_posi - score_nega) #self.ratio * (score_posi_binary - score_nega_binary)
        return (score_posi, score_nega, item_codes_text_posi, item_codes_img_posi)
        #return (output.view(-1, 1).squeeze(1), output_binary)

    def accuracy(self, output=None, target=None):
        """Compute the current accuracy."""
        correct = torch.gt(output.data, 0).sum()
        res = float(correct) / float(output.data.numel())
        return res
