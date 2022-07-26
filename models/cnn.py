from email.mime import base
import os
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, 3), 1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, (3, 3), 1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(output_channel)

    def forward(self, input):
        x_in = self.conv1(input)
        x_hid = self.batch_norm1(x_in)
        x_hid = self.relu(x_hid)
        x_hid = self.conv2(x_hid)
        x_hid = self.batch_norm2(x_hid)
        x_out = x_in + x_hid
        return self.relu(x_out)


class ResNetBlock3D(nn.Module):

    def __init__(self, input_channel, output_channel, downsamp=False):
        super().__init__()
        self.downsamp = downsamp
        if self.downsamp: downsamp_stride = 2
        else: downsamp_stride = 1

        self.conv1 = nn.Conv3d(input_channel, output_channel, kernel_size=3, stride=downsamp_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(output_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(output_channel)
        if self.downsamp:
            self.downsample = nn.Sequential(
                nn.Conv3d(input_channel, output_channel, kernel_size=1, stride=downsamp_stride, bias=False), 
                nn.BatchNorm3d(output_channel)
            )

    def forward(self, inputs):
        x = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(inputs)))))
        if self.downsamp: inputs = self.downsample(inputs)
        outputs = x + inputs
        return outputs


class ResNetLayer3D(nn.Module):

    def __init__(self, input_channel, output_channel, num_blocks, first_layer):
        super().__init__()
        self.block_list = nn.Sequential()

        if first_layer: assert input_channel == output_channel
        
        for i in range(num_blocks):
            if i == 0:
                if first_layer: block = ResNetBlock3D(input_channel, input_channel, downsamp=False)
                else: block = ResNetBlock3D(input_channel, output_channel, downsamp=True)
            else:
                block = ResNetBlock3D(output_channel, output_channel)
            self.block_list.append(block)

    def forward(self, inputs):
        x = inputs
        x = self.block_list(x)
        return x


class ResNet3D(nn.Module):

    def __init__(self, input_channel, hidden_num_channel, num_classes, layer_structure):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channel, hidden_num_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_num_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.layer_list = nn.Sequential()
        for i, num_block in enumerate(layer_structure):
            if i==0: layer = ResNetLayer3D(hidden_num_channel, hidden_num_channel, num_block, True)
            else: 
                layer = ResNetLayer3D(hidden_num_channel, hidden_num_channel*2, num_block, False)
                hidden_num_channel *= 2
            self.layer_list.append(layer)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.linear = nn.Linear(2048, 1, bias=True)
        self.activation = torch.nn.Sigmoid()

    def forward(self, inputs):
        x = inputs
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer_list(x)
        
        x = self.avgpool(x)
        print(f"Avgpool: {x.size()}")
        x = x.view(1, -1)
        print(f"Linear: {x.size()}")
        x = self.linear(x)
        output = self.activation(x)
        return output


class ResNet(nn.Module):

    def __init__(self, input_channel):
        super().__init__()
        self.input_channel = input_channel
        self.conv1 = nn.Conv2d(input_channel, 32, (3, 3), 1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.res_block1 = ResNetBlock(32, 32)
        self.res_block2 = ResNetBlock(32, 32)
        self.res_block3 = ResNetBlock(32, 64)
        #self.down_conv1 = nn.Conv2d(64, 64, (3, 3), 2, padding=2)
        self.down_conv1 = nn.MaxPool2d(2, stride=2)
        self.res_block4 = ResNetBlock(64, 64)
        self.res_block5 = ResNetBlock(64, 64)
        self.res_block6 = ResNetBlock(64, 64)
        self.res_block7 = ResNetBlock(64, 64)
        self.res_block8 = ResNetBlock(64, 64)
        self.res_block9 = ResNetBlock(64, 128)
        #self.down_conv2 = nn.Conv2d(128, 128, (3, 3), 2, padding=2)
        self.down_conv2 = nn.MaxPool2d(2, stride=2)
        self.res_block10 = ResNetBlock(128, 128)
        self.res_block11 = ResNetBlock(128, 128)
        self.res_block12 = ResNetBlock(128, 128)
        self.res_block13 = ResNetBlock(128, 128)
        self.res_block14 = ResNetBlock(128, 128)
        self.res_block15 = ResNetBlock(128, 256)
        #self.down_conv3 = nn.Conv2d(256, 256, (3, 3), 2, padding=2)
        self.down_conv3 = nn.MaxPool2d(2, stride=2)
        self.res_block16 = ResNetBlock(256, 256)
        self.res_block17 = ResNetBlock(256, 256)
        self.res_block18 = ResNetBlock(256, 512)
        #self.down_conv4 = nn.Conv2d(512, 512, (3, 3), 2, padding=2)
        self.down_conv4 = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(512, 1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.down_conv1(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        x = self.res_block9(x)
        x = self.down_conv2(x)
        x = self.res_block10(x)
        x = self.res_block11(x)
        x = self.res_block12(x)
        x = self.res_block13(x)
        x = self.res_block14(x)
        x = self.res_block15(x)
        x = self.down_conv3(x)
        x = self.res_block16(x)
        x = self.res_block17(x)
        x = self.res_block18(x)
        x = self.down_conv4(x)
        x = torch.squeeze(self.avg_pool(x))
        output = self.dense(x)
        return output

class Linear(nn.Module):
    def __init__(self, input_dim, activation="sigmoid"):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        if activation=="sigmoid":
            self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)

def get_pretrained_resnet(name_model="resnet50", input_channels=3):
    """
    Get a pretrained ResNet
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', name_model, pretrained=True)
    if input_channels != 3:
        weights = model.conv1.weight.clone()
        model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight[:, :3] = weights
            model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
    output_layer = Linear(2048)
    model.fc = output_layer
    return model


def save_network_config(model, file_name=None):

    if not(file_name):
        base_dir = "/mnt/beegfs/home/phuc/my-code/dsc-predict/model-summary/"
        num_run = len(os.listdir(base_dir))
        txt_file_dir = os.path.join(base_dir, f"run-{num_run+1}.txt")
    else:
        txt_file_dir = file_name
    with open(txt_file_dir, "w") as f:

        # Print out the structure of the model
        f.write("-------------------------------------\nStructure of model: \n")
        print(model, file=f)

        # Print out the weights of the model
        sum_weights = 0
        f.write("-------------------------------------\n\nWeights of model: \n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                sum_weights += param.numel()
                f.write(f"{name}: {param.numel()}\n")
        f.write(f"-------------------------------------\n\nTotal number of params: {sum_weights}\n")


if __name__ == "__main__":

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')

    #model = get_pretrained_resnet("resnet34")
    model = ResNet3D(4, 64, 1, [3, 4, 6, 3]).to(device)
    #pytorch_total_params = [p.numel() for p in model.parameters() if p.requires_grad]
    #print(sum(pytorch_total_params))
    #print(model.fc)
    save_network_config(model, "temp2.txt")

    ran_tensor = torch.rand((4, 4, 128, 128, 128)).to(device)
    output = model(ran_tensor)
    print(f"Output: {output.size()}")

