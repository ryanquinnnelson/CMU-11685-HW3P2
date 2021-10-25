from torch import nn
import torch

class XLCTCmodel(nn.Module):
    def __init__(self, **kwargs):
        super(XLCTCmodel,self).__init__()
        # TODO: Your model goes here

        raise NotImplementedError

        self.conv = None
        self.lstm =
        self.fc =
        self.relu =
        self.drop =
        self.pack = nn.utils.rnn.pack_padded_sequence
        self.unpack = nn.utils.rnn.pad_packed_sequence

    def forward(self, x, length):
        # x: BxLxC
        x = x.permute(0, 2, 1)
        # x: BxCxL
        if self.conv is not None:
            x = self.conv(x)
        # x: BxCxL
        x = x.permute(2, 0, 1)
        # x: LxBxC
        x = self.pack(x, length, batch_first=False, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = self.unpack(x, batch_first=False)
        # x: LxBxC
        x = x.permute(1, 0, 2)
        # x: BLxC
        B, L, C = x.shape
        #x = x.flatten(0, 1)
        x = self.drop(x)
        x = self.fc(x)
        # x: BxLxC
        x = x.view(B, L, 42)
        x = x.permute(1, 0, 2)
        x = x.log_softmax(2)
        return x