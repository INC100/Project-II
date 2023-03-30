import torch

#使用鉴别器判断图片是真是假，与target比较，计算torch.nn.MSELoss
def identify_image(discriminator, image, target):
    output = discriminator(image)
    loss = torch.nn.MSELoss()
    return loss(output, target)