import torch.nn as nn
import torch
from dataset_utils import center_crop

class UNet3D_Mini(nn.Module):

    def downward_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        The downward blocks are part of the contracting path.
        Utility:
        - decrease the depth, height and width.
        - increase the number of channels.
        The result is that features get more high level
        and spatial information decreases with each downward block.
        """
        downward_block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),

            nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=0),
            nn.ReLU()
        )
        return downward_block

    def upward_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        The upward blocks are part of the expanding path.
        Utility:
        - increase the depth, height and width.
        - decrease the number of channels.
        The result is a segmentation map that increases
        in resolution for each upward block.
        """

        upward_block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),

            nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

        return upward_block

    def __init__(self, num_modalities, num_classes):

        self.DEBUG = False

        super(UNet3D_Mini, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # downward layers
        self.down1 = self.downward_block(in_channels=num_modalities, out_channels=64,
                                         kernel_size=3, stride=1, padding=0)

        self.down2 = self.downward_block(in_channels=64, out_channels=128,
                                         kernel_size=3, stride=1, padding=0)

        self.down3 = self.downward_block(in_channels=128, out_channels=256,
                                         kernel_size=3, stride=1, padding=1)


        self.upconv1 = nn.ConvTranspose3d(in_channels=256, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, output_padding=0)

        self.up1 = self.upward_block(in_channels=256, out_channels=128,
                                     kernel_size=3, stride=1, padding=1)

        self.upconv2 = nn.ConvTranspose3d(in_channels=128, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, output_padding=0)

        self.up2 = self.upward_block(in_channels=128, out_channels=64,
                                     kernel_size=3, stride=1, padding=1)

        self.out_conv = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=num_classes,
                      kernel_size=3, stride=1, padding=1),
            #nn.ReLU(),
        )

    def forward(self, x):

        # downward layers
        x = self.down1(x)
        l1_out = torch.clone(x)
        l1_crop_shape = (torch.tensor(l1_out.shape) - torch.tensor([0,0,16,16,16])).tolist()
        l1_out = center_crop(l1_crop_shape[2], l1_crop_shape[3], l1_crop_shape[4], l1_out)
        x = self.maxpool(x)

        x = self.down2(x)
        l2_out = torch.clone(x)
        l2_crop_shape = (torch.tensor(l2_out.shape) - torch.tensor([0,0,4,4,4])).tolist()
        l2_out = center_crop(l2_crop_shape[2], l2_crop_shape[3], l2_crop_shape[4], l2_out)
        x = self.maxpool(x)

        x = self.down3(x)
        l3_out = torch.clone(x)


        if self.DEBUG:
            print('l1_out:', l1_out.shape)
            #print('l1_out_crop:', l1_out_crop.shape)
            print('l2_out:', l2_out.shape)
            #print('l2_out_crop:', l2_out_crop.shape)
            print('l3_out:', l3_out.shape)
            #print('l3_out_crop:', l3_out_crop.shape)
            #print('l4_out:', l4_out.shape)
            print('x:', x.shape)

        # upward layers
        x = self.upconv1(x)
        if self.DEBUG:
            print('after upconv1:', x.shape)
        x = torch.cat((l2_out, x), dim=1)
        if self.DEBUG:
            print('after cat:', x.shape)
        x = self.up1(x)
        if self.DEBUG:
            print('after up1:', x.shape)

        x = self.upconv2(x)
        if self.DEBUG:
            print('after upconv2:', x.shape)
        x = torch.cat((l1_out, x), dim=1)
        if self.DEBUG:
            print('after cat:', x.shape)
        x = self.up2(x)
        if self.DEBUG:
            print('after up2:', x.shape)

        # output conv block
        x = self.out_conv(x)
        if self.DEBUG:
            print('after out conv:', x.shape)
        return x
