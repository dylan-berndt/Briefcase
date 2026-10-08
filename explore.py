from utils import *
import plotly.graph_objects as go
import plotly


location = "levjepa_levjepa-d256-e200_step38500"
method = "TSNE"
outputName = "fontMap_levjepa_tsne.html"
keepSources = ("google", "dafont")    # None keeps every source in the .npz


if __name__ == "__main__":
    if not os.path.exists(os.path.join("embeddings", f"{location}.json")):
        model, config = ViT.load(os.path.join("checkpoints", "pretrain", "latest"))
        dataset = CombinedQueryData(config.dataset, training=False)

        embeddings = generateEmbeddings(
            {"names": dataset.names,
            "paths": dataset.paths,
            "letters": dataset.letters},
            model = model,
            fileName = f"{location}"
        )
    else:
        with open(os.path.join("embeddings", f"{location}.json"), "r") as file:
            embeddings = json.load(file)
    print("Compressing... ")
    allKeys = list(embeddings)

    # fonts with no glyphs in the cache were left as zero vectors
    embeddings = {key: value for key, value in embeddings.items() if any(value)}

    # only the sources the web search uses; the .npz rows are in the same order as the .json keys
    if keepSources is not None:
        with np.load(os.path.join("embeddings", f"{location}.npz")) as npz:
            sourceOf = dict(zip(allKeys, npz["sources"]))
        embeddings = {key: value for key, value in embeddings.items() if sourceOf[key] in keepSources}
        print(f"{len(embeddings)} fonts from {keepSources}")
    compressed = compressEmbeddings(embeddings, components=6, method=method, gpu=True)

    with open(os.path.join("results", "fontPaths.txt"), "r") as pathFile:
        paths = pathFile.read().split("\n")
        path = paths[0].split(" -> ")

    x = np.stack([compressed[key][0] for key in compressed], axis=0)
    y = np.stack([compressed[key][1] for key in compressed], axis=0)
    z = np.stack([compressed[key][2] for key in compressed], axis=0)
    labels = list(compressed.keys())

    r = np.stack([compressed[key][3] for key in compressed], axis=0)
    g = np.stack([compressed[key][4] for key in compressed], axis=0)
    b = np.stack([compressed[key][5] for key in compressed], axis=0)

    r, g, b, = 255 * (r - r.min()) / (r.max() - r.min()), 255 * (g - g.min()) / (g.max() - g.min()), 255 * (b - b.min()) / (b.max() - b.min())

    colors = [f"rgb({int(r[i])}, {int(g[i])}, {int(b[i])})" for i in range(len(labels))]

    fig = go.Figure(
        data = [go.Scatter3d(x=x, y=y, z=z, mode="markers", text=labels, hovertemplate="%{text}<extra></extra>", marker=dict(size=5, color=colors))],
        layout=go.Layout(template="plotly_dark")
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False
            ),
            zaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                visible=False
            )
        )
    )

    lineX = []
    lineY = []
    lineZ = []

    for a, b in zip(path[:-1], path[1:]):
        ax, ay, az = compressed[a][:3]
        bx, by, bz = compressed[b][:3]

        lineX.extend([ax, bx, None])
        lineY.extend([ay, by, None])
        lineZ.extend([az, bz, None])

    fig.write_html(os.path.join("results", outputName))

    # fig.add_trace(
    #     go.Scatter3d(x=lineX, y=lineY, z=lineZ, mode="lines", line=dict(width=4, color="red"), hoverinfo="none")
    # )

    fig.show()

    