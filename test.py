from model import *
from data import *
import pygame
import math

testPath = os.path.join("checkpoints", "latest")
modelPath = os.path.join(testPath, "checkpoint.pt")
configPath = os.path.join(testPath, "config.json")

config = Config().load(configPath)
model = UNet(config.model)
model.load_state_dict(torch.load(modelPath, weights_only=False).state_dict())
model.eval()

pygame.init()
windowSize = [960, 640]
window = pygame.display.set_mode(windowSize)
clock = pygame.time.Clock()

imageSize = 48
scale = 7

canvas = np.zeros((imageSize, imageSize), dtype=np.bool)

drawing = False
drawingMode = "add"
lx, ly = (None, None)
cursorSize = 2

classColumns = 52
uiFont = pygame.font.SysFont("Calibri", 12)

uiSurface = pygame.Surface(windowSize, pygame.SRCALPHA, 32)
uiSurface = uiSurface.convert_alpha()

gridWidth = (windowSize[0] - 120) // classColumns
gridHeight = (windowSize[1] - 180 - imageSize * scale) / math.ceil(len(characters) / classColumns)
for c, char in enumerate(characters):
    column = c % classColumns
    row = c // classColumns
    position = 120 + imageSize * scale + row * gridHeight, 60 + column * gridWidth
    text = uiFont.render(char, False, [255, 255, 255])
    uiSurface.blit(text, (position[1], position[0]))


while True:
    window.fill((55, 75, 200))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            if event.button == 3:
                drawingMode = "sub"
            else:
                drawingMode = "add"
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        # Good luck
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                mx, my = pygame.mouse.get_pos()
                cx, cy = mx - 60, my - 60
                dx, dy = mx - lx, my - ly
                distance = max(int(math.sqrt(dx ** 2 + dy ** 2)), 1)
                for i in range(distance):
                    path = i / distance
                    fx, fy = cx + dx * path, cy + dy * path
                    nx, ny = int(fx // scale), int(fy // scale)
                    canvas[nx: nx + cursorSize, ny: ny + cursorSize] = drawingMode == "add"

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFTBRACKET:
                cursorSize -= 1
            if event.key == pygame.K_RIGHTBRACKET:
                cursorSize += 1

    lx, ly = pygame.mouse.get_pos()

    sdf = dist(canvas) - dist(~canvas)
    inputs = torch.tensor(sdf / imageSize, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    if config.dataset.maps == "bitmaps":
        inputs = torch.tensor(canvas.transpose(), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        output, classification = model(inputs)
        choice = nn.functional.softmax(classification, dim=-1).squeeze().detach().numpy()
        output = output.squeeze().detach().numpy().transpose()

    out1 = np.clip((output - output.min()) / (output.max() - output.min() + 1e-8), 0, 1)
    out = output > 0
    out = out.astype(np.uint8) * 255

    if config.dataset.maps == "bitmaps":
        out = out1.astype(np.float32) * 255

    image = canvas.astype(np.uint8) * 255
    img = pygame.surfarray.make_surface(np.stack([image] * 3, axis=-1))
    img = pygame.transform.scale(img, (imageSize * scale, imageSize * scale))
    window.blit(img, (60, 60))

    img = pygame.surfarray.make_surface(np.stack([out] * 3, axis=-1))
    img = pygame.transform.scale(img, (imageSize * scale, imageSize * scale))
    window.blit(img, (windowSize[0] - imageSize * scale - 60, 60))

    gridWidth = (windowSize[0] - 120) // classColumns
    gridHeight = (windowSize[1] - 180 - imageSize * scale) / math.ceil(len(characters) / classColumns)
    for c, char in enumerate(characters):
        column = c % classColumns
        row = c // classColumns
        position = 120 + imageSize * scale + row * gridHeight, 60 + column * gridWidth
        pygame.draw.rect(window, [255 * choice[c], 0, 0], [position[1], position[0], gridWidth, gridHeight])
    
    window.blit(uiSurface, [0, 0])

    pygame.display.flip()
    clock.tick(60)

