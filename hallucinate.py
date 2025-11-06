from utils import *
import numpy as np
from itertools import product
from PIL import Image

testPath = os.path.join("checkpoints", "pretrain", "upper")
model, config = UNet.load(testPath)
model.eval()

characters = [chr(c) for c in range(ord('a'), ord('z') + 1)]

imageSize = 48
highScores = {target: 100000 for target in characters}
best = {target: np.zeros([imageSize, imageSize]) for target in characters}

objective = nn.CrossEntropyLoss()


def centerLoss(image):
    gx, gy = torch.meshgrid(torch.arange(image.shape[1]), torch.arange(image.shape[1]), indexing="ij")
    gx, gy = gx / image.shape[1], gy / image.shape[1]
    # print(gx.shape, gy.shape, gx.mean(), gy.mean(), (gx.unsqueeze(1) * image).shape, (gx.unsqueeze(1) * image).mean())
    xCenter = (gx.unsqueeze(-1) * image).sum() / image.sum()
    yCenter = (gy.unsqueeze(-1) * image).sum() / image.sum()
    xSpread = ((gx.unsqueeze(-1) - xCenter) ** 2 * image).sum() / image.sum()
    ySpread = ((gy.unsqueeze(-1) - xCenter) ** 2 * image).sum() / image.sum()
    # loss = torch.pow(xSpread, 2) + torch.pow(ySpread, 2)
    loss = (xCenter - 0.5) ** 2 + (yCenter - 0.5) ** 2
    loss = 1 * loss + 0.6 * (xSpread + ySpread)
    return loss


# Holy cow that's a lot of images
for i, char in enumerate(characters):
    progress = 0

    imagePath = os.path.join("data", "bitmaps", f"Calibri Regular {char}l.bmp")
    img = np.fromfile(imagePath, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    image = img.astype(np.float32) / 255.0
    # image = np.random.uniform(0, 1, (imageSize, imageSize))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    image.requires_grad_(True)
    optimizer = torch.optim.Adam([image], 0.1)
    while highScores[char] > -100 or torch.mean(image).item() > 0.3:
        optimizer.zero_grad()
        outputs, classification = model(image)

        fillLoss = (torch.mean(image) - 0.2) ** 2
        binaryReg = torch.mean(image * (1 - image))
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, 1:, :, :] - image[:, :-1, :, :]
        noiseLoss = torch.mean(dx ** 2) + torch.mean(dy ** 2)
        centering = centerLoss(image)
        # characterScore = objective(classification, torch.tensor([i], dtype=torch.long))
        characterScore = -(classification[0, i] - classification[0].mean())
        loss = characterScore + 0.6 * binaryReg + 0.8 * fillLoss + 0.6 * noiseLoss + 0.6 * centering
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            image.clamp_(0, 1)

        if loss.item() < highScores[char]:
            highScores[char] = loss.item()
            best[char] = image.detach().squeeze().numpy()

        progress += 1
        print(f"\r{progress} iterations of {char} | Score: {highScores[char]:.4f} | Character: {characterScore.item():.3f} | Fill: {torch.mean(image).item():.3f} | Noise: {noiseLoss.item():.3f} | Centering: {centering.item():.3f}", end="")
        if progress > 600:
            highScores[char] = loss.item()
            best[char] = image.detach().squeeze().numpy()
            break

    im = Image.fromarray((best[char] * 255).astype(np.uint8))
    im.save(os.path.join("oblivion", char + ".bmp"))

    print()
