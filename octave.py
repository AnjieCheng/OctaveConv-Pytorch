import torch.nn as nn

class firstOctConv(nn.Module):
    def __init__(self, settings, ch_in, ch_out, kernel=(1,1), pad=(0,0), stride=(1,1)):
        super(firstOctConv, self).__init__()
        self.stride = stride
        _, alpha_out = settings

        hf_ch_out = int(ch_out * (1 - alpha_out))
        lf_ch_out = ch_out - hf_ch_out

        self.hf_ch_out = hf_ch_out
        self.lf_ch_out = lf_ch_out

        if stride == (2, 2):
            self.downsample = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=True)
        self.hf_conv = nn.Conv2d(ch_in, hf_ch_out, kernel_size=kernel, stride=(1,1), padding=pad, bias=False)
        self.hf_pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=True)
        self.hf_pool_conv = nn.Conv2d(ch_in, lf_ch_out, kernel_size=kernel, stride=(1,1), padding=pad, bias=False)

    def forward(self, x):
        if self.stride== (2, 2):
            x = self.downsample(x)

        out_h = self.hf_conv(x)
        x = self.hf_pool(x)
        out_l = self.hf_pool_conv(x)
        return out_h, out_l

class lastOctConv(nn.Module):
    def __init__(self, settings, ch_in, ch_out, kernel=(1,1), pad=(0,0), stride=(1,1)):
        super(lastOctConv, self).__init__()
        self.stride = stride
        alpha_in, alpha_out = settings
        hf_ch_in = int(ch_in * (1 - alpha_in))
        hf_ch_out = int(ch_out * (1 - alpha_out))

        lf_ch_in = ch_in - hf_ch_in #TODO: Check here!

        self.hf_ch_out = hf_ch_out

        if stride == (2, 2):
            self.downsample = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=True)
        self.hf_conv = nn.Conv2d(hf_ch_in, hf_ch_out, kernel_size=kernel, stride=(1,1), padding=pad, bias=False)
        # self.lf_conv = nn.Conv2d(lf_ch_in, hf_ch_out, kernel_size=kernel, stride=(1,1), padding=pad, bias=False)

    def forward(self, hf_data, lf_data):
        if self.stride== (2, 2):
            x = self.downsample(x)

        out_h = self.hf_conv(hf_data)
        # out_l = self.lf_conv(lf_data)
        return out_h #+out_l

class OctConv(nn.Module):
    def __init__(self, settings, ch_in, ch_out, kernel=(1,1), pad=(0,0), stride=(1,1)):
        super(OctConv, self).__init__()
        self.stride = stride
        alpha_in, alpha_out = settings
        hf_ch_in = int(ch_in * (1 - alpha_in))
        hf_ch_out = int(ch_out * (1 - alpha_out))

        lf_ch_in = ch_in - hf_ch_in
        lf_ch_out = ch_out - hf_ch_out

        self.hf_ch_out = hf_ch_out
        self.lf_ch_out = lf_ch_out

        if stride == (2, 2):
            self.downsample = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=True)
        self.hf_conv = nn.Conv2d(hf_ch_in, hf_ch_out, kernel_size=kernel, stride=(1,1), padding=pad, bias=False)
        self.hf_pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=True)
        self.hf_pool_conv = nn.Conv2d(hf_ch_in, lf_ch_out, kernel_size=kernel, stride=(1,1), padding=pad, bias=False)

        self.lf_conv = nn.Conv2d(lf_ch_in, hf_ch_out, kernel_size=kernel, stride=(1,1), padding=pad, bias=False)
        if stride == (2, 2):
            self.lf_down = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=True)
        else:
            self.lf_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lf_down_conv = nn.Conv2d(lf_ch_in, lf_ch_out, kernel_size=kernel, stride=(1,1), padding=pad, bias=False)

    def forward(self, hf_data, lf_data):
        if self.stride == (2, 2):
            hf_data = self.downsample(hf_data)
        hf_conv = self.hf_conv(hf_data)
        hf_pool = self.hf_pool(hf_data)
        hf_pool_conv = self.hf_pool_conv(hf_pool)

        lf_conv = self.lf_conv(lf_data)
        if self.stride== (2, 2):
            lf_upsample = lf_conv
            lf_down = self.lf_down(lf_data)
        else:
            lf_upsample = self.lf_upsample(lf_conv)
            lf_down = lf_data
        lf_down_conv = self.lf_down_conv(lf_down)

        out_h = hf_conv + lf_upsample
        out_l = hf_pool_conv + lf_down_conv

        return out_h, out_l


class firstOctConv_BN_AC(nn.Module):
    def __init__(self, alpha, num_filter_in, num_filter_out, kernel, pad, stride=(1,1), num_group=1):
        super(firstOctConv_BN_AC, self).__init__()
        self.conv = firstOctConv(settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out, kernel=kernel, pad=pad, stride=stride)
        self.hf_bn = nn.BatchNorm2d(self.conv.hf_ch_out)
        self.lf_bn = nn.BatchNorm2d(self.conv.lf_ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hf_data, lf_data = self.conv(x)
        out_hf = self.hf_bn(hf_data)
        out_hf = self.relu(out_hf)
        out_lf = self.hf_bn(lf_data)
        out_lf = self.relu(out_lf)
        return out_hf, out_lf

class lastOctConv_BN_AC(nn.Module):
    def __init__(self, alpha, num_filter_in, num_filter_out, kernel, pad, stride=(1,1), num_group=1):
        super(lastOctConv_BN_AC, self).__init__()
        self.conv = lastOctConv(settings=(alpha, 0), ch_in=num_filter_in, ch_out=num_filter_out, kernel=kernel, pad=pad, stride=stride)
        self.bn = nn.BatchNorm2d(self.conv.hf_ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hf_data, lf_data):
        out = self.conv(hf_data, lf_data)
        out = self.bn(out)
        out = self.relu(out)
        return out


class octConv_BN_AC(nn.Module):
    def __init__(self, alpha, num_filter_in, num_filter_out, kernel, pad, stride=(1,1), num_group=1):
        super(octConv_BN_AC, self).__init__()
        self.conv = OctConv(settings=(alpha, alpha), ch_in=num_filter_in, ch_out=num_filter_out, kernel=kernel, pad=pad, stride=stride)
        self.hf_bn = nn.BatchNorm2d(self.conv.hf_ch_out)
        self.lf_bn = nn.BatchNorm2d(self.conv.lf_ch_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hf_data, lf_data):
        hf_data, lf_data = self.conv(hf_data, lf_data)
        out_hf = self.hf_bn(hf_data)
        out_hf = self.relu(out_hf)
        out_lf = self.hf_bn(lf_data)
        out_lf = self.relu(out_lf)
        return out_hf, out_lf


class firstOctConv_BN(nn.Module):
    def __init__(self, alpha, num_filter_in, num_filter_out, kernel, pad, stride=(1,1), num_group=1):
        super(firstOctConv_BN, self).__init__()
        self.conv = firstOctConv(settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out, kernel=kernel, pad=pad, stride=stride)
        self.hf_bn = nn.BatchNorm2d(self.conv.hf_ch_out)
        self.lf_bn = nn.BatchNorm2d(self.conv.lf_ch_out)

    def forward(self, x):
        hf_data, lf_data = self.conv(x)
        out_hf = self.hf_bn(hf_data)
        out_lf = self.hf_bn(lf_data)
        return out_hf, out_lf


class lastOctConv_BN(nn.Module):
    def __init__(self, alpha, num_filter_in, num_filter_out, kernel, pad, stride=(1,1), num_group=1):
        super(lastOctConv_BN, self).__init__()
        self.conv = lastOctConv(settings=(alpha, 0), ch_in=num_filter_in, ch_out=num_filter_out, kernel=kernel, pad=pad, stride=stride)
        self.bn = nn.BatchNorm2d(self.conv.hf_ch_out)

    def forward(self, hf_data, lf_data):
        out = self.conv(hf_data, lf_data)
        out = self.bn(out)
        return out

class octConv_BN(nn.Module):
    def __init__(self, alpha, num_filter_in, num_filter_out, kernel, pad, stride=(1,1), num_group=1):
        super(octConv_BN, self).__init__()
        self.conv = OctConv(settings=(alpha, alpha), ch_in=num_filter_in, ch_out=num_filter_out, kernel=kernel, pad=pad, stride=stride)
        self.hf_bn = nn.BatchNorm2d(self.conv.hf_ch_out)
        self.lf_bn = nn.BatchNorm2d(self.conv.lf_ch_out)

    def forward(self, hf_data, lf_data):
        hf_data, lf_data = self.conv(hf_data, lf_data)
        out_hf = self.hf_bn(hf_data)
        out_lf = self.hf_bn(lf_data)
        return out_hf, out_lf

class Conv_BN(nn.Module):
    def __init__(self, num_filter_in, num_filter_out, kernel, pad, stride=(1,1)):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(num_filter_in, num_filter_out, kernel_size=kernel, padding=pad, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(num_filter_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# class Residual_Unit_norm(nn.Module):
#     def __init__(self, num_in, num_mid, num_out, first_block=False, stride=(1, 1), g=1):
#         self.conv_m1 = Conv_BN_AC(num_filter=num_mid, kernel=(1, 1), pad=(0, 0))
#         self.conv_m2 = Conv_BN_AC(num_filter=num_mid, kernel=(3, 3), pad=(1, 1), stride=stride, num_group=g)
#         self.conv_m3 = Conv_BN(num_filter=num_out, kernel=(1, 1), pad=(0, 0))

#         if first_block:
#             self.shortcut = Conv_BN(num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=stride)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.conv_m1(x)
#         out = self.conv_m2(x)
#         out = self.conv_m3(x)
#         if first_block:
#             x = self.shortcut(x)
#         out = out + x
#         return self.relu(out)

class Residual_Unit_last(nn.Module):
    def __init__(self, alpha, num_in, num_mid, num_out, first_block=False, stride=(1, 1), g=1):
        super(Residual_Unit_last, self).__init__()
        self.conv_m1 = octConv_BN_AC(alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid, kernel=(1, 1), pad=(0, 0))
        self.conv_m2 = lastOctConv_BN_AC(alpha=alpha, num_filter_in=num_mid, num_filter_out=num_mid, kernel=(3,3), pad=(1,1), stride=stride)
        self.conv_m3 = Conv_BN(num_filter_in=num_mid, num_filter_out=num_out, kernel=(1, 1), pad=(0, 0))
        self.first_block = first_block
        # if first_block:
        self.shortcut = lastOctConv_BN(alpha=alpha, num_filter_in=num_in, num_filter_out=num_out, kernel=(1,1), pad=(0,0), stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hf_data, lf_data):
        hf_data_m, lf_data_m = self.conv_m1(hf_data, lf_data)
        conv_m2 = self.conv_m2(hf_data_m, lf_data_m)
        conv_m3 = self.conv_m3(conv_m2)
        # if self.first_block:
        x = self.shortcut(hf_data, lf_data)
        out = conv_m3 + x
        return self.relu(out)

class Residual_Unit_first(nn.Module):
    def __init__(self, alpha, num_in, num_mid, num_out, first_block=False, stride=(1, 1), g=1):
        super(Residual_Unit_first, self).__init__()
        self.conv_m1 = firstOctConv_BN_AC(alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid, kernel=(1, 1), pad=(0, 0))
        self.conv_m2 = octConv_BN_AC(alpha=alpha, num_filter_in=num_mid, num_filter_out=num_mid, kernel=(3,3), pad=(1,1), stride=stride)
        self.conv_m3 = octConv_BN(alpha=alpha, num_filter_in=num_mid, num_filter_out=num_out, kernel=(1, 1), pad=(0, 0))
        self.first_block = first_block
        if first_block:
            self.shortcut = firstOctConv_BN(alpha=alpha, num_filter_in=num_in, num_filter_out=num_out, kernel=(1,1), pad=(0,0), stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        hf_data_m, lf_data_m = self.conv_m1(x)
        hf_data_m, lf_data_m = self.conv_m2(hf_data_m, lf_data_m)
        hf_data_m, lf_data_m = self.conv_m3(hf_data_m, lf_data_m)
        if self.first_block:
            hf_data, lf_data = self.shortcut(x)

        hf_outputs = hf_data + hf_data_m
        lf_outputs = lf_data + lf_data_m

        return self.relu(hf_outputs), self.relu(lf_outputs)


class Residual_Unit(nn.Module):
    def __init__(self, alpha, num_in, num_mid, num_out, first_block=False, stride=(1, 1), g=1):
        super(Residual_Unit, self).__init__()
        self.conv_m1 = octConv_BN_AC(alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid, kernel=(1, 1), pad=(0, 0))
        self.conv_m2 = octConv_BN_AC(alpha=alpha, num_filter_in=num_mid, num_filter_out=num_mid, kernel=(3,3), pad=(1,1), stride=stride)
        self.conv_m3 = octConv_BN(alpha=alpha, num_filter_in=num_mid, num_filter_out=num_out, kernel=(1, 1), pad=(0, 0))
        self.first_block = first_block
        if first_block:
            self.shortcut = octConv_BN(alpha=alpha, num_filter_in=num_in, num_filter_out=num_out, kernel=(1,1), pad=(0,0), stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, hf_data, lf_data):
        hf_data_m, lf_data_m = self.conv_m1(hf_data, lf_data)
        hf_data_m, lf_data_m = self.conv_m2(hf_data_m, lf_data_m)
        hf_data_m, lf_data_m = self.conv_m3(hf_data_m, lf_data_m)
        if self.first_block:
            hf_data, lf_data = self.shortcut(hf_data, lf_data)

        hf_outputs = hf_data + hf_data_m
        lf_outputs = lf_data + lf_data_m

        return self.relu(hf_outputs), self.relu(lf_outputs)
