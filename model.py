import torch
import torch.nn as nn
from utils import patchify

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as T
import torchvision
import io
import os

from utils import get_positional_embeddings

class MyViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()

        # attributes
        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])

        # learnable classification token
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # positional embedding
        self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
        self.pos_embed.requires_grad = False

        # transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # classification MLP
        self.mlp = nn.Sequential(
                nn.Linear(self.hidden_d, out_d),
                nn.Softmax(dim=-1)
                )


    def forward(self, images):
        n, c, h, w = images.shape
        # divide images into patches
        patches = patchify(images, self.n_patches)

        # map the vector corresponding to each dimension to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # adding classification token to the tokens
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # adding positional embedding
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed

        # transformer blocks
        for block in self.blocks:
            out = block(out)

        # getting the classification token only
        out = out[:, 0]

        return self.mlp(out)


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []

        for sequence in sequences:
            seq_result = []

            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)

        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
                nn.Linear(hidden_d, mlp_ratio * hidden_d),
                nn.GELU(),
                nn.Linear(mlp_ratio * hidden_d, hidden_d)
                )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


# main
if __name__ == "__main__":
    model = MyViT(chw=(1, 140, 140))

    image = Image.open("test_img.jpeg")
    transform = T.Compose([T.Resize((140, 140)),
        T.Grayscale(),
        ToTensor()])
    x = transform(image)
    x = x.unsqueeze(0)

    print("Original size: ", image.size)
    x_img = ToPILImage()(x.squeeze(0))
    x_img.show()
    print("Input size: ", x.size())

    out = model(x)
    print("Out size: ", out.size())

#    patches.squeeze_(0)
#    patches.resize_(49, 20, 20)
#    patches.unsqueeze_(1)
#
#    print(patches.size())
#
#    grid_tensor = torchvision.utils.make_grid(patches, nrow=7)
#    print(f"Grid tensor shape: {grid_tensor.shape}")
#
#    grid_img = T.ToPILImage()(grid_tensor)
#    print(f"Grid img shape: {grid_img.size}")
#
#    grid_img.show()
#
#if __name__ == "__main__":
#    import matplotlib.pyplot as plt
#    emb = get_positional_embeddings(100, 300)
#    plt.imshow(emb, cmap="hot", interpolation="nearest")
#    plt.show()
