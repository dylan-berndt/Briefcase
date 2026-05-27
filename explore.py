from utils import *
import plotly.graph_objects as go


if __name__ == "__main__":
    with open(os.path.join("embeddings", "google.json"), "r") as file:
        embeddings = json.load(file)
    compressed = compressEmbeddings(embeddings, components=3)

    with open(os.path.join("results", "fontPaths.txt"), "r") as pathFile:
        paths = pathFile.read().split("\n")
        path = paths[0].split(" -> ")

    x = np.stack([compressed[key][0] for key in compressed], axis=0)
    y = np.stack([compressed[key][1] for key in compressed], axis=0)
    z = np.stack([compressed[key][2] for key in compressed], axis=0)
    labels = list(compressed.keys())

    fig = go.Figure(
        data = [go.Scatter3d(x=x, y=y, z=z, mode="markers", text=labels, hovertemplate="%{text}<extra></extra>", marker=dict(size=5))]
    )

    lineX = []
    lineY = []
    lineZ = []

    for a, b in zip(path[:-1], path[1:]):
        ax, ay, az = compressed[a]
        bx, by, bz = compressed[b]

        lineX.extend([ax, bx, None])
        lineY.extend([ay, by, None])
        lineZ.extend([az, bz, None])

    fig.write_html(os.path.join("results", "fontMap.html"))

    fig.add_trace(
        go.Scatter3d(x=lineX, y=lineY, z=lineZ, mode="lines", line=dict(width=4, color="red"), hoverinfo="none")
    )

    fig.show()

    