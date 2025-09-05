from model import *
import numpy as np
from itertools import product
from PIL import Image

testPath = os.path.join("checkpoints", "latest")
modelPath = os.path.join(testPath, "checkpoint.pt")
configPath = os.path.join(testPath, "config.json")

config = Config().load(configPath)
model = UNet(config.model)
model.load_state_dict(torch.load(modelPath, weights_only=False).state_dict())
model.eval()

characters = [chr(c) for c in range(ord('a'), ord('z') + 1)]

imageSize = 48
highScores = {target: 100000 for target in characters}
best = {target: np.zeros([imageSize, imageSize]) for target in characters}

objective = nn.CrossEntropyLoss()

# Holy cow that's a lot of images
for i, char in enumerate(characters):
    progress = 0
    image = np.random.rand(imageSize, imageSize)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    image.requires_grad_(True)
    optimizer = torch.optim.Adam([image], 0.01)
    while highScores[char] > 0.07 or torch.mean(image).item() > 0.3:
        optimizer.zero_grad()
        outputs, classification = model(image)
        score = nn.functional.softmax(classification, dim=-1).squeeze()[i]
        fillLoss = (torch.mean(image) - 0.15) ** 2
        binaryReg = torch.mean(image * (1 - image))
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, 1:, :, :] - image[:, :-1, :, :]
        noiseLoss = torch.mean(dx ** 2) + torch.mean(dy ** 2)
        loss = objective(classification, torch.tensor([i], dtype=torch.long)) + 0.4 * binaryReg + 0.2 * fillLoss + 0.8 * noiseLoss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            image.clamp_(0, 1)

        if loss.item() < highScores[char]:
            highScores[char] = loss.item()
            best[char] = image.detach().squeeze().numpy()

        progress += 1
        print(f"\r{progress} iterations of {char} | Score: {highScores[char]} | Fill: {torch.mean(image).item()}", end="")

    im = Image.fromarray((best[char] * 255).astype(np.uint8))
    im.save(os.path.join("hallucinations", char + ".bmp"))

    print()
