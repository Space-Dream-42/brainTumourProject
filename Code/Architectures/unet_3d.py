import torch.nn as nn
import torch

class UNet3D(nn.Module):

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
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
        return downward_block

    def upward_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        The upward blocks are part of the expanding path.
        Utility:
        - increase the number of depth, height and width.
        - decrease the number of channels.
        The result is a segmentation map that increases
        in resolution for each upward block.
        """
        mid_channels = in_channels // 2

        upward_block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=mid_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),

            nn.Conv3d(in_channels=mid_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )

        return upward_block

    def __init__(self, num_modalities, num_classes):

        self.DEBUG = False

        super(UNet3D, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # downward layers
        self.down1 = self.downward_block(in_channels=num_modalities, out_channels=8,
                                         kernel_size=3, stride=1, padding=1)

        self.down2 = self.downward_block(in_channels=8, out_channels=8,
                                         kernel_size=3, stride=1, padding=1)

        self.down3 = self.downward_block(in_channels=8, out_channels=16,
                                         kernel_size=3, stride=1, padding=1)

        self.same_conv = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=16,
                                          kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up1 = self.upward_block(in_channels=32, out_channels=16,
                                     kernel_size=3, stride=1, padding=0)

        self.upconv2 = nn.ConvTranspose3d(in_channels=16, out_channels=8,
                                          kernel_size=10, stride=2, padding=0, output_padding=0)

        self.up2 = self.upward_block(in_channels=16, out_channels=8,
                                     kernel_size=3, stride=1, padding=0)

        self.upconv3 = nn.ConvTranspose3d(in_channels=8, out_channels=num_classes,
                                          kernel_size=9, stride=2, padding=0, output_padding=1)

        self.up3 = self.upward_block(in_channels=12, out_channels=num_classes,
                                     kernel_size=3, stride=1, padding=1)

        self.out_conv = nn.Sequential(
            nn.Conv3d(in_channels=num_classes, out_channels=num_classes,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):

        # downward layers
        x = self.down1(x)
        l1_out = torch.clone(x)
        x = self.maxpool(x)

        x = self.down2(x)
        l2_out = torch.clone(x)
        x = self.maxpool(x)

        x = self.down3(x)
        l3_out = torch.clone(x)
        x = self.maxpool(x)

        # bottom floor
        x = self.same_conv(x)
        x = self.same_conv(x)

        if self.DEBUG:
            print('l1_out:', l1_out.shape)
            print('l2_out:', l2_out.shape)
            print('l3_out:', l3_out.shape)
            print('x:', x.shape)

        # upward layers
        x = self.upconv1(x)
        if self.DEBUG:
            print('after upconv1:', x.shape)
        x = torch.cat((l3_out, x), dim=1)
        if self.DEBUG:
            print('after cat:', x.shape)
        x = self.up1(x)
        if self.DEBUG:
            print('after up1:', x.shape)

        x = self.upconv2(x)
        if self.DEBUG:
            print('after upconv2:', x.shape)
        x = torch.cat((l2_out, x), dim=1)
        if self.DEBUG:
            print('after cat:', x.shape)
        x = self.up2(x)
        if self.DEBUG:
            print('after up2:', x.shape)

        x = self.upconv3(x)
        if self.DEBUG:
            print('after upconv3:', x.shape)
        x = torch.cat((l1_out, x), dim=1)
        if self.DEBUG:
            print('after cat:', x.shape)
        x = self.up3(x)
        if self.DEBUG:
            print('after up3:', x.shape)

        # output conv block
        x = self.out_conv(x)
        if self.DEBUG:
            print('after out conv:', x.shape)
        return x




class Small_UNet3D(nn.Module):

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
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )
        return downward_block

    def upward_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        The upward blocks are part of the expanding path.
        Utility:
        - increase the number of depth, height and width.
        - decrease the number of channels.
        The result is a segmentation map that increases
        in resolution for each upward block.
        """
        mid_channels = in_channels // 2

        upward_block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=mid_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm3d(mid_channels),
            nn.Conv3d(in_channels=mid_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )

        return upward_block

    def __init__(self, num_modalities, num_classes):

        self.DEBUG = False

        super(Small_UNet3D, self).__init__()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # downward layers
        self.down1 = self.downward_block(in_channels=num_modalities, out_channels=8,
                                         kernel_size=3, stride=1, padding=1)

        self.down2 = self.downward_block(in_channels=8, out_channels=8,
                                         kernel_size=3, stride=1, padding=1)

        self.down3 = self.downward_block(in_channels=8, out_channels=8,
                                         kernel_size=3, stride=1, padding=1)

        self.same_conv = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose3d(in_channels=8, out_channels=8,
                                          kernel_size=3, stride=2, padding=1, output_padding=1)

        self.up1 = self.upward_block(in_channels=16, out_channels=8,
                                     kernel_size=2, stride=1, padding=0)

        self.upconv2 = nn.ConvTranspose3d(in_channels=8, out_channels=8,
                                          kernel_size=5, stride=2, padding=0, output_padding=1)

        self.up2 = self.upward_block(in_channels=16, out_channels=8,
                                     kernel_size=3, stride=1, padding=0)

        self.upconv3 = nn.ConvTranspose3d(in_channels=8, out_channels=8,
                                          kernel_size=9, stride=2, padding=0, output_padding=1)

        self.up3 = self.upward_block(in_channels=16, out_channels=num_classes,
                                     kernel_size=3, stride=1, padding=1)

        self.out_conv = nn.Sequential(
            nn.Conv3d(in_channels=num_classes, out_channels=num_classes,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):

        # downward layers
        x = self.down1(x)
        l1_out = torch.clone(x)
        x = self.maxpool(x)

        x = self.down2(x)
        l2_out = torch.clone(x)
        x = self.maxpool(x)

        x = self.down3(x)
        l3_out = torch.clone(x)
        x = self.maxpool(x)

        # bottom floor
        x = self.same_conv(x)

        if self.DEBUG:
            print('l1_out:', l1_out.shape)
            print('l2_out:', l2_out.shape)
            print('l3_out:', l3_out.shape)
            print('x:', x.shape)

        # upward layers
        x = self.upconv1(x)
        if self.DEBUG:
            print('after upconv1:', x.shape)
        x = torch.cat((l3_out, x), dim=1)
        if self.DEBUG:
            print('after cat:', x.shape)
        x = self.up1(x)
        if self.DEBUG:
            print('after up1:', x.shape)

        x = self.upconv2(x)
        if self.DEBUG:
            print('after upconv2:', x.shape)
        x = torch.cat((l2_out, x), dim=1)
        if self.DEBUG:
            print('after cat:', x.shape)
        x = self.up2(x)
        if self.DEBUG:
            print('after up2:', x.shape)

        x = self.upconv3(x)
        if self.DEBUG:
            print('after upconv3:', x.shape)
        x = torch.cat((l1_out, x), dim=1)
        if self.DEBUG:
            print('after cat:', x.shape)
        x = self.up3(x)
        if self.DEBUG:
            print('after up3:', x.shape)

        # output conv block
        x = self.out_conv(x)
        if self.DEBUG:
            print('after out conv:', x.shape)
        del l1_out, l2_out, l3_out
        return x