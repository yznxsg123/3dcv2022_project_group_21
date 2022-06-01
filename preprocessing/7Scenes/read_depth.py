import matplotlib.pyplot as plt 
from PIL import Image
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.ndimage import map_coordinates as Warp
import torch
import torch.nn.functional as F
from torchvision import transforms

# img = plt.imread('livingroom1-depth-clean/00000.png')
# plt.imsave('depth_00.png', img, cmap='jet')
# img = plt.imread('livingroom1-depth-clean/00020.png')
# plt.imsave('depth_20.png', img, cmap='jet')
# exit()
'''
# AICL_NUIM
img0 = Image.open('livingroom1-color/00000.jpg')
depth0 = Image.open('livingroom1-depth-clean/00000.png').convert('I')
img1 = Image.open('livingroom1-color/00020.jpg')
depth1 = Image.open('livingroom1-depth-clean/00020.png').convert('I')
'''
# ICL_NUIM
img0 = Image.open('office/seq-01/frame-000000.color.png')
depth0 = Image.open('office/seq-01/frame-000000.depth.png').convert('I')
img1 = Image.open('office/seq-01/frame-000020.color.png')
depth1 = Image.open('office/seq-01/frame-000020.depth.png').convert('I')

pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()
img0 = pil2tensor(img0).unsqueeze(0) #[1, 3, H, W]
img1 = pil2tensor(img1).unsqueeze(0) #[1, 3, H, W]
depth1 = pil2tensor(depth1).double().unsqueeze(0) #[1, 3, H, W]

'''
# AICL_NUIM
dataset = 'AICL'
pose0 = np.array([[-0.2739592186924325, 0.021819345900466677, -0.9614937663021573, -0.31057997014702826],
                    [8.33962904204855e-19, -0.9997426093226981, -0.02268733357278151, 0.5730122438481298],
                    [-0.9617413095492113, -0.006215404179813816, 0.27388870414358013, 2.1264800183565487],
                    [0, 0, 0, 1 ]])
pose1 = np.array([[-0.18638284464153276, 0.25695841490084653, -0.9482793935518026, -0.30253456363671116],
                    [-6.580010103117446e-18, -0.9651922712131064, -0.2615413534997576, 1.0465825339569201],
                    [-0.9824771932331714, -0.04874682145668151, 0.1798952811347205, 2.1103388549508497],
                    [0, 0, 0, 1 ]])
'''
# ICL_NUIM
dataset = 'ICL'
pose0 = np.array([[  9.3306959e-001,	 -1.6964546e-001,	  3.1708300e-001,	 -9.4861656e-001	],
[  1.7337367e-001,	  9.8468649e-001,	  1.6643673e-002,	 -5.6749225e-001],	
[ -3.1506166e-001,	  3.9447054e-002,	  9.4821417e-001,	  7.1117169e-001],
[0, 0, 0, 1]])
pose1 = np.array([[  9.1686714e-001,	 -2.3301134e-001,	  3.2403532e-001,	 -9.5679641e-001	],
[  2.2543879e-001,	  9.7229767e-001,	  6.1283499e-002,	 -6.0436767e-001	],
[ -3.2935017e-001,	  1.6863447e-002,	  9.4401991e-001,	  7.7693993e-001],
[0, 0, 0, 1]])
# K = np.array([[481.20, 0, 319.5],
#                 [0, 480, 239.5],
#                 [0, 0, 1]])
K = np.array([[585.0, 0, 320],
                [0, 585, 240],
                [0, 0, 1]

])
p_1_0 = torch.from_numpy(np.matmul(np.linalg.inv(pose0), pose1)).unsqueeze(0)

K = torch.from_numpy(K)
K_inv = K.inverse().double()
# generate_uv_coord
b, _, h, w = img1.shape

u_range, v_range = torch.linspace(0, w-1, w), torch.linspace(0, h-1, h)
grid_v, grid_u = torch.meshgrid([v_range, u_range])
grid_ones = torch.ones(h, w)
uv_coords = torch.stack((grid_u, grid_v, grid_ones), dim=0) # [3, H, W]
uv_coords = uv_coords.expand(b, *uv_coords.size()).double() # [B, 3, H, W]

# generate_warp_coords
depth_scale = 1000
cam_coords = torch.matmul(K_inv, uv_coords.reshape(b, 3, -1))
cam_coords = cam_coords * depth1.reshape(b, 1, h*w).expand(b, 3, h*w) / depth_scale
translation = p_1_0[:, :3, 3].unsqueeze(-1)  # [B, 3, 1]
rot_mat = p_1_0[:, :3, :3] # [B, 3, 3]
proj_coords = torch.matmul(rot_mat, cam_coords) + translation
warp_coords = torch.matmul(K, proj_coords) # [B, 3, H*W]
src_x, src_y, src_z = warp_coords[:, 0], warp_coords[:, 1], warp_coords[:, 2].clamp(min=1e-10) # [B, H*W]
# print warp_coords
src_u = 2 * (src_x / src_z) / (w - 1) - 1
src_v = 2 * (src_y / src_z) / (h - 1) - 1
warp_coords = torch.stack([src_u, src_v], dim=2)  # [B, H*W, 2]
warp_coords = warp_coords.reshape(b, h, w, 2)
warped_img0 = F.grid_sample(img0.double(), warp_coords, padding_mode='zeros')


output = tensor2pil(warped_img0[0].float())
img = np.array(output, dtype=np.uint8)
plt.imshow(np.array(output))
plt.imsave(dataset + '_warped_from_0_to_20_scale_%d.png'%depth_scale, img)
# plt.show()
# plt.imshow(tensor2pil(img0[0]))
plt.show()
