import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# ===== NeRF Model =====
class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )

        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid()
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma


# ===== Data Preparation =====
def prepare_dataset_from_image(image_path, camera_intrinsics, camera_pose):
    image = cv2.imread(image_path)
    H, W, _ = image.shape
    image = image / 255.0  # Normalize to [0, 1]

    fx, fy, cx, cy = camera_intrinsics
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    pixel_coords = np.stack([i, j, np.ones_like(i)], axis=-1)  # [H, W, 3]
    K_inv = np.linalg.inv(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    ray_directions = pixel_coords @ K_inv.T
    ray_directions /= np.linalg.norm(ray_directions, axis=-1, keepdims=True)
    ray_directions = (camera_pose[:3, :3] @ ray_directions.reshape(-1, 3).T).T
    ray_origins = np.tile(camera_pose[:3, 3], (H * W, 1))
    colors = image.reshape(-1, 3)
    dataset = np.concatenate([ray_origins, ray_directions, colors], axis=1)
    return torch.tensor(dataset, dtype=torch.float32)


# ===== Rendering Rays =====
def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=1, nb_bins=192):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)
    return c + 1 - weight_sum.unsqueeze(-1)


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


# ===== Training =====
def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_bins=192, nb_epochs=10):
    nerf_model.train()
    for epoch in tqdm(range(nb_epochs)):
        for batch in data_loader:
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()


# ===== Testing =====
@torch.no_grad()
def render_novel_view(nerf_model, camera_intrinsics, camera_pose, H, W, hn=0, hf=1, nb_bins=192, device='cpu'):
    dataset = prepare_dataset_from_image("scene_image.jpg", camera_intrinsics, camera_pose)
    ray_origins = dataset[:, :3].to(device)
    ray_directions = dataset[:, 3:6].to(device)

    data = []
    for i in range(0, ray_origins.shape[0], 1024):
        origins_chunk = ray_origins[i:i + 1024]
        directions_chunk = ray_directions[i:i + 1024]
        data.append(render_rays(nerf_model, origins_chunk, directions_chunk, hn, hf, nb_bins))
    img = torch.cat(data, dim=0).cpu().numpy()
    return img.reshape(H, W, 3)


# ===== Main Workflow =====
if __name__ == "__main__":
    # Input settings
    image_path = "1.png"  # Path to your input image
    camera_intrinsics = [800, 800, 400, 400]  # fx, fy, cx, cy (example values)
    camera_pose = np.eye(4)  # Example camera pose (identity matrix)

    # Prepare dataset
    dataset = prepare_dataset_from_image(image_path, camera_intrinsics, camera_pose)
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # Initialize model and training components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NerfModel(hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)

    # Train the model
    train(model, optimizer, scheduler, data_loader, device=device, hn=2, hf=6, nb_bins=192, nb_epochs=16)

    # Render a novel view
    novel_camera_pose = np.eye(4)  # Change for a new perspective
    H, W = 400, 400
    output_image = render_novel_view(model, camera_intrinsics, novel_camera_pose, H, W, hn=2, hf=6, nb_bins=192, device=device)

    # Save the output
    plt.imshow(output_image)
    plt.axis('off')
    plt.savefig("novel_view.png")
    plt.show()
