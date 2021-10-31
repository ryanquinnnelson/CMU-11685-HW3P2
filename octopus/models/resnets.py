"""
All things Resnet.
"""

__author__ = 'ryanquinnnelson'

import torch.nn as nn


def _init_weights(layer):
    """
    Perform initialization of layer weights if layer is a Conv2d layer.
    Args:
        layer: layer under consideration

    Returns: None

    """
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)


class ResidualBlock(nn.Module):
    """
    Define a standard residual block of a Resnet. Takes inspiration from
    https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            # initialize to kaiming normal
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # blocks
        out = self.blocks(x)

        if self.in_channels != self.out_channels:
            # shortcut
            shortcut = self.shortcut(x)

            # combine
            activate = nn.ReLU(inplace=True)
            out = activate(out + shortcut)

        return out


class ResidualBlock2(nn.Module):
    """
    Define a standard residual block of a Resnet.

    The difference from ResidualBlock is that ResidualBlock2 uses Identity layers for the shortcut rather than skipping
    the shortcut when it does not need to change output dimension. This improves learning rate quite a bit for an
     unknown reason.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        # blocks
        out = self.blocks(x)

        # shortcut
        shortcut = self.shortcut(x)

        # combine
        out = self.activate(out + shortcut)

        return out


class ResidualBlock3(nn.Module):
    """
    Defines a residual block in a Resnet.

    This is the same as ResidualBlock2, except it includes kaiming initialization of all CNN layers.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = 2 if in_channels != out_channels else 1

        # first conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)

        # second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.blocks = nn.Sequential(

            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            self.conv2,
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        # blocks
        out = self.blocks(x)

        # shortcut
        shortcut = self.shortcut(x)

        # combine
        out = self.activate(out + shortcut)

        return out


class ResidualBlock4(nn.Module):
    """
    Defines a residual block for Resnet.

    Same as ResidualBlock3 except it includes kaiming initialization in a way that doesn't introduce twice the number
    of layers when viewing model summary
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.activate = nn.ReLU(inplace=True)

        # initialize weights
        self.blocks.apply(_init_weights)
        self.shortcut.apply(_init_weights)

    def forward(self, x):
        # blocks
        out = self.blocks(x)

        # shortcut
        shortcut = self.shortcut(x)

        # combine
        out = self.activate(out + shortcut)

        return out


class Resnet18(nn.Module):
    """
    Implements Resnet18. Takes inspiration from
    https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278

    Leaves kernel_size,stride,and padding of first CNN layer as defined in the Resnet paper (7,2,3).
    Includes extra linear layers for use in extracting the feature embedding from the model.
    """

    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            # conv3..x
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),

            # conv4..x
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),

            # conv5..x
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))
        # nn.Softmax(dim=1))  # removed because it stopped model from improving

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        """
        Execute forward pass on model using data and either return the output or the output and embedding.
        Args:
            x (Tensor): batch of data
            return_embedding (Boolean): True to return embedding as well as model output.

        Returns: Tensor representing model output if return_embedding=False,
        Tuple (embedding,output) if return_embedding=True.

        """
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


class Resnet34(nn.Module):
    """
    Implements Resnet34.

    Leaves kernel_size,stride,and padding of first CNN layer as defined in the Resnet paper (7,2,3).
    Includes extra linear layers for use in extracting the feature embedding from the model.
    """

    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            # conv3..x
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),

            # conv4..x
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            # conv5..x
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        """
        Execute forward pass on model using data and either return the output or the output and embedding.
        Args:
            x (Tensor): batch of data
            return_embedding (Boolean): True to return embedding as well as model output.

        Returns: Tensor representing model output if return_embedding=False,
        Tuple (embedding,output) if return_embedding=True.

        """
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


class Resnet34_v2(nn.Module):
    """
    Implements Resnet34. Same as Resnet34 except it uses ResidualBlock2.
    """

    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock2(64, 64),
            ResidualBlock2(64, 64),
            ResidualBlock2(64, 64),

            # conv3..x
            ResidualBlock2(64, 128),
            ResidualBlock2(128, 128),
            ResidualBlock2(128, 128),
            ResidualBlock2(128, 128),

            # conv4..x
            ResidualBlock2(128, 256),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256),
            ResidualBlock2(256, 256),

            # conv5..x
            ResidualBlock2(256, 512),
            ResidualBlock2(512, 512),
            ResidualBlock2(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        """
        Execute forward pass on model using data and either return the output or the output and embedding.
        Args:
            x (Tensor): batch of data
            return_embedding (Boolean): True to return embedding as well as model output.

        Returns: Tensor representing model output if return_embedding=False,
        Tuple (embedding,output) if return_embedding=True.

        """
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


class Resnet34_v3(nn.Module):
    """
    Implements Resnet34. Same as Resnet34_v2 except it uses ResidualBlock3 and changes kernel_size,stride to 3,1
    instead of 7,2 from original implementation in order to work better with the smaller images in a 64 x 64 dimension
    dataset. Includes kaiming initialization for first CNN layer. Removes the max
    pool layer from the original implementation.
    """

    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        # conv1
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.layers = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock3(64, 64),
            ResidualBlock3(64, 64),
            ResidualBlock3(64, 64),

            # conv3..x
            ResidualBlock3(64, 128),
            ResidualBlock3(128, 128),
            ResidualBlock3(128, 128),
            ResidualBlock3(128, 128),

            # conv4..x
            ResidualBlock3(128, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),

            # conv5..x
            ResidualBlock3(256, 512),
            ResidualBlock3(512, 512),
            ResidualBlock3(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        """
        Execute forward pass on model using data and either return the output or the output and embedding.
        Args:
            x (Tensor): batch of data
            return_embedding (Boolean): True to return embedding as well as model output.

        Returns: Tensor representing model output if return_embedding=False,
        Tuple (embedding,output) if return_embedding=True.

        """
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


class Resnet34_v4(nn.Module):
    """
    Implements Resnet34. Same as Resnet34_v3 except it uses ResidualBlock4. Moves kaiming initialization for first CNN
    layer to a function to prevent duplicate layers from showing up in model summary.
    """
    def __init__(self, in_features, num_classes, feat_dim=512):
        super().__init__()
        self.feat_dim = feat_dim

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock4(64, 64),
            ResidualBlock4(64, 64),
            ResidualBlock4(64, 64),

            # conv3..x
            ResidualBlock4(64, 128),
            ResidualBlock4(128, 128),
            ResidualBlock4(128, 128),
            ResidualBlock4(128, 128),

            # conv4..x
            ResidualBlock4(128, 256),
            ResidualBlock4(256, 256),
            ResidualBlock4(256, 256),
            ResidualBlock4(256, 256),
            ResidualBlock4(256, 256),
            ResidualBlock4(256, 256),

            # conv5..x
            ResidualBlock4(256, 512),
            ResidualBlock4(512, 512),
            ResidualBlock4(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

        # initialize weights
        self.layers.apply(_init_weights)

    def forward(self, x, return_embedding=False):
        """
        Execute forward pass on model using data and either return the output or the output and embedding.
        Args:
            x (Tensor): batch of data
            return_embedding (Boolean): True to return embedding as well as model output.

        Returns: Tensor representing model output if return_embedding=False,
        Tuple (embedding,output) if return_embedding=True.

        """
        embedding = self.layers(x)
        embedding_out = self.activation(self.linear_feat_dim(embedding))
        output = self.linear(embedding)

        if return_embedding:
            return embedding_out, output
        else:
            return output


class Resnet34_v5(nn.Module):
    """
    Implements Resnet34. Same as Resnet34_v3 except it modifies forward() to use Flatten() layer for embedding. This is
    a workaround to load trained models that were initialized with feat_dim=2 (because it can't be easily changed
    afterward).
    """
    def __init__(self, in_features, num_classes, feat_dim=2):
        super().__init__()
        self.feat_dim = feat_dim

        # conv1
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=3, stride=1, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.layers = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            ResidualBlock3(64, 64),
            ResidualBlock3(64, 64),
            ResidualBlock3(64, 64),

            # conv3..x
            ResidualBlock3(64, 128),
            ResidualBlock3(128, 128),
            ResidualBlock3(128, 128),
            ResidualBlock3(128, 128),

            # conv4..x
            ResidualBlock3(128, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),
            ResidualBlock3(256, 256),

            # conv5..x
            ResidualBlock3(256, 512),
            ResidualBlock3(512, 512),
            ResidualBlock3(512, 512),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(512, num_classes))

        self.linear_feat_dim = nn.Linear(512, self.feat_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, return_embedding=False):
        """
        Execute forward pass on model using data and either return the output or the output and embedding.
        Args:
            x (Tensor): batch of data
            return_embedding (Boolean): True to return embedding as well as model output.

        Returns: Tensor representing model output if return_embedding=False,
        Tuple (embedding,output) if return_embedding=True.

        """
        embedding = self.layers(x)
        output = self.linear(embedding)

        if return_embedding:
            return embedding, output
        else:
            return output


class BottleneckResidualBlock(nn.Module):
    """
    Defines a residual block that involves bottlenecking, for use in larger Resnet models.
    """
    def __init__(self, in_channels, out_channels, use_shortcut, mid_stride, shortcut_stride):
        super().__init__()

        self.use_shortcut = use_shortcut

        self.stride = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(

            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),

            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=mid_stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),

            # third conv layer
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True)
        )

        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=shortcut_stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):

        # blocks
        out = self.blocks(x)

        if self.use_shortcut:
            # shortcut
            shortcut = self.shortcut(x)

            # combine
            activate = nn.ReLU(inplace=True)
            out = activate(out + shortcut)

        return out


class Resnet50(nn.Module):
    """
    Implements Resnet50.

    Leaves kernel_size,stride,and padding of first CNN layer as defined in the Resnet paper (7,2,3).
    Includes extra linear layers for use in extracting the feature embedding from the model.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            BottleneckResidualBlock(64, 64, use_shortcut=True, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv3..x
            BottleneckResidualBlock(256, 128, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv4..x
            BottleneckResidualBlock(512, 256, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv5..x
            BottleneckResidualBlock(1024, 512, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(2048, num_classes))

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)


class Resnet101(nn.Module):
    """
    Implements Resnet101.

    Leaves kernel_size,stride,and padding of first CNN layer as defined in the Resnet paper (7,2,3).
    Includes extra linear layers for use in extracting the feature embedding from the model.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            BottleneckResidualBlock(64, 64, use_shortcut=True, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv3..x
            BottleneckResidualBlock(256, 128, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv4..x
            BottleneckResidualBlock(512, 256, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv5..x
            BottleneckResidualBlock(1024, 512, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(2048, num_classes))

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)


class Resnet152(nn.Module):
    """
    Implements Resnet152.

    Leaves kernel_size,stride,and padding of first CNN layer as defined in the Resnet paper (7,2,3).
    Includes extra linear layers for use in extracting the feature embedding from the model.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # conv2..x
            BottleneckResidualBlock(64, 64, use_shortcut=True, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(256, 64, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv3..x
            BottleneckResidualBlock(256, 128, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(512, 128, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv4..x
            BottleneckResidualBlock(512, 256, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(1024, 256, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # conv5..x
            BottleneckResidualBlock(1024, 512, use_shortcut=True, mid_stride=2, shortcut_stride=2),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),
            BottleneckResidualBlock(2048, 512, use_shortcut=False, mid_stride=1, shortcut_stride=1),

            # summary
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        # decoding layer
        self.linear = nn.Sequential(
            nn.Linear(2048, num_classes))

    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)

        if return_embedding:
            return embedding
        else:
            return self.linear(embedding)
