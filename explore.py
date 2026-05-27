from utils import *
import plotly.graph_objects as go


if __name__ == "__main__":
    model, config = ViT.load(os.path.join("checkpoints", "pretrain", "latest"))
    data = collectFontSetPaths("google", config.dataset.fontSize, "bitmaps")

    embeddings = generateEmbeddings(data, model)
    compressed = compressEmbeddings(embeddings, components=3)

    x = np.stack([compressed[key][:, 0] for key in compressed], axis=0)
    y = np.stack([compressed[key][:, 1] for key in compressed], axis=0)
    z = np.stack([compressed[key][:, 2] for key in compressed], axis=0)
    labels = list(compressed.keys())

    fig = go.Figure(
        data = [go.Scatter3d(x=x, y=y, z=z, mode="markers", text=labels, hovertemplate="%{text}<extra></extra>", marker=dict(size=5))]
    )

    fig.show()