import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img

def train(D, G, optim_D, optim_G, discriminator_loss, generator_loss, show_every=250, 
              batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_loader:
             
            
            optim_D.zero_grad()
            real_images = preprocess_img(x).to(device)
            logits_real = D(real_images)
            Random_Noise = Variable(sample_noise(batch_size, noise_size))
            Random_Noise=Random_Noise.to(device)
            fake_images = G(Random_Noise).detach()
            logits_fake = D(fake_images.view(batch_size, 3, 64, 64))
            d_total_error = discriminator_loss(logits_real, logits_fake).mean()
            d_total_error.backward()        
            optim_D.step()
            optim_G.zero_grad()
            Random_Noise = Variable(sample_noise(batch_size, noise_size))
            Random_Noise=Random_Noise.to(device)
            fake_images = G(Random_Noise)
            Fake_Noise = D(fake_images.view(batch_size, 3, 64, 64))
            g_error = generator_loss(Fake_Noise).mean()
            g_error.backward()
            optim_G.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
                imgs_numpy = fake_images.data.cpu().numpy()
                
                show_images(imgs_numpy[0:16],color=True)
                
                plt.show()
                print()
            iter_count += 1