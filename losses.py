import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.autograd import Variable
dtype=dtype = torch.FloatTensor
def discriminator_loss(logits_real, logits_fake):
    loss = None
    N = logits_real.size()
    true_labels = Variable(torch.ones(N)).type(dtype)
    real_image_loss = bce_loss(logits_real.cpu(), true_labels)
    fake_image_loss = bce_loss(logits_fake.cpu(), 1 - true_labels)
    loss = real_image_loss + fake_image_loss
    return loss

def generator_loss(logits_fake):
    loss = None
    N = logits_fake.size()
    true_labels = Variable(torch.ones(N)).type(dtype)
    loss = bce_loss(logits_fake.cpu(), true_labels)
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    loss = None
    N = scores_real.size()
    loss_real = 0.5*torch.mean(torch.pow(scores_real.cpu()-Variable(torch.ones(N)).type(dtype), 2))
    loss_fake = 0.5*torch.mean(torch.pow(scores_fake.cpu(), 2))
    loss = loss_real + loss_fake
    return loss

def ls_generator_loss(scores_fake):
    loss = None
    N = scores_fake.size()
    loss = 0.5*torch.mean(torch.pow(scores_fake.cpu()-Variable(torch.ones(N)).type(dtype), 2))
    return loss
