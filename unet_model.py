"""
Defines the U-Net model architecture.
Classes:
- DoubleConv: Applies two sets of convolution, batch normalization, and ReLU activation.
- Down:
    Downscales the input using max pooling followed by a double convolution.
- Up: 
    Upscales the input using transposed convolution or bilinear interpolation followed by a double convolution.
- OutConv:
    Final convolution layer to produce the output with the desired number of channels.
- UNet:
    The main U-Net architecture that combines the above components to create a complete model for image segmentation tasks.

Some comments asking and answering what the code does but not looking into why the architecture is designed this way.

"""
import torch.nn.functional as F
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    applies (convolution -> BatchNorm -> ReLU) * 2
    """
# why do we need mid_channels?
# to have more flexibility between the two layers
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            # what does nn.Sequential do?
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x): 
        # input x is a tensor of shape (batch_size, in_channels, height, width)
        # output will be a tensor of shape (batch_size, out_channels, height, width)
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling using maxpool then applying double conv module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """
    Upscaling the feature map using transposed convolution followed by double conv module
    """
    # what is transpose convolution?
    # A transposed convolution (also known as deconvolution or fractionally strided convolution) is a type of convolution that increases the spatial dimensions of the input feature map.
    # how does it do that?
    # It does this by applying a convolution operation that effectively reverses the downsampling process, allowing the model to learn how to upsample the feature maps.

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear is True, use the normal convolutions to reduce the number of channels
        # but what is bilinear?
        # Bilinear interpolation is a method of resampling that uses linear interpolation in two dimensions, which is often used for upsampling images.
        # why do we use bilinear interpolation?
        # Bilinear interpolation is used to maintain spatial resolution and smoothness in the upsampled feature maps, which is important for tasks like image segmentation.
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # what does align_corners do?
            # The align_corners parameter in nn.Upsample determines how the corner pixels of the input and output tensors are aligned during upsampling.
            # It ensures that the corner pixels of the input tensor are aligned with the corner pixels of the output tensor, which can help preserve spatial relationships in the feature maps.
            # note: in_channels // 2 is used to reduce the number of channels after upsampling
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the input tensor of shape (batch_size, in_channels, height, width)
        # x2 is the skip connection tensor of shape (batch_size, out_channels, height, width)
        # how come we are using skip connections?
        # Skip connections are used to combine features from earlier layers with features from later layers, allowing the model to retain spatial information and improve gradient flow during training.
        
        # previous classes had a simple forward method, but here we have two inputs 
        # why is that?
        # The forward method in the Up class takes two inputs: x1 (the upsampled feature map) and x2 (the skip connection feature map).
        # what does the upsample do?
        # The upsample method increases the spatial dimensions of x1 by a factor of 2, effectively doubling its height and width.
        # After upsampling, we concatenate x1 and x2 along the channel dimension to combine the features from both layers.
        # Then, we apply the convolutional layers to the concatenated tensor to produce the final output.

        x1 = self.up(x1)
        # CHW input
        # get difference in height
        diffY = x2.size()[2] - x1.size()[2]
        # get difference in width
        diffX = x2.size()[3] - x1.size()[3]
        # pad the upsampled tensor to match the size of the skip connection tensor
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # concatenate the upsampled tensor and the skip connection tensor along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        # apply the convolutional layers to the concatenated tensor
        return self.conv(x)
    
class OutConv(nn.Module):
    """Final convolution layer to produce the output with the desired number of channels."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        # how come there are arguments in the super() here, when earlier it was super().__init__()?
        # The arguments in `super(UNet, self).__init__()` specify the class and instance for the superclass initialization, which is necessary for proper inheritance in Python 2.x. In Python 3.x, you can simply use `super().__init__()`, but using the full form is still valid and works in both versions.
        # but in the same code why two diferent ways of calling super? for DoubleConv etc different and for UNet different?
        # In Python 3.x, you can use `super().__init__()` without arguments, which is more concise and preferred. The `super(UNet, self).__init__()` syntax is more common in Python 2.x or when you want to be explicit about the class and instance, but it's not necessary in Python 3.x.
        # if its not necessary then why do it?
        # It is often done for compatibility with older codebases or to maintain consistency in style, especially in mixed Python 2.x and 3.x environments. However, in modern Python code (Python 3.x), using `super().__init__()` is the recommended approach for simplicity and readability.
        # what would happen if I did it the regular way here?
        # If you used `super().__init__()` in the `UNet` class, it would work perfectly fine in Python 3.x, as it automatically refers to the current class and instance. However, if you were using Python 2.x or wanted to maintain compatibility with both versions, you would need to use the full syntax `super(UNet, self).__init__()`. In modern Python code, using `super().__init__()` is preferred for its simplicity.
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # why is it called inc()?
        # inc() is a common naming convention for the initial convolutional layer in U-Net architectures.
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # how do we know if bilinear is True or False? 
        # The `bilinear` parameter is typically set when creating an instance of the UNet class, allowing the user to choose whether to use bilinear interpolation for upsampling or not.
        # why do we need to divide by 2?
        # The division by 2 in the `Down` class is used to reduce the number of channels after downsampling, effectively halving the number of channels at each downsampling step. This helps to control the model's complexity and memory usage while retaining important features.
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x is the input tensor of shape (batch_size, n_channels, height, width)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits is the output tensor of shape (batch_size, n_classes, height, width)
        return logits
    

