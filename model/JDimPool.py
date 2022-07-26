import torch.nn as nn

class JDimPool(nn.MaxPool1d):
    def forward(self, input):
        b,n, c, w, h = input.size()
        input = input.view(b, n, c* w * h).permute(0, 2, 1)
        pooled = nn.functional.max_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(b,c, w, h)