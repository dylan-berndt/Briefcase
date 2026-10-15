import json

import pytest

from mapspecimens import MapSpecimens

IMAGES = [b"RIFF-one", None, b"RIFF-three-longer"]


def writeMap(directory, name="routes", images=IMAGES):
    pack, entries = b"", []
    for image in images:
        if image is None:
            entries.append(None)
        else:
            entries.append([len(pack), len(image)])
            pack += image
    (directory / f"{name}.pack").write_bytes(pack)
    (directory / f"{name}.index.json").write_text(json.dumps({"count": len(images), "specimens": entries}))


def test_a_point_gets_its_own_image_and_a_missing_one_gets_none(tmp_path):
    writeMap(tmp_path)
    specimens = MapSpecimens(str(tmp_path))
    assert specimens.get("routes", 0)[0] == b"RIFF-one"
    assert specimens.get("routes", 2)[0] == b"RIFF-three-longer"
    assert specimens.get("routes", 1) is None                     # a point with no specimen
    assert specimens.get("routes", 3) is None and specimens.get("routes", -1) is None
    assert specimens.get("nowhere", 0) is None


def test_no_folder_just_means_no_images(tmp_path):
    assert MapSpecimens(str(tmp_path / "missing")).get("routes", 0) is None


def test_an_unpulled_lfs_pointer_is_refused(tmp_path):
    writeMap(tmp_path, images=[b"x" * 500, None, b"y" * 500])      # real images are far bigger than a pointer file
    (tmp_path / "routes.pack").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="git-lfs"):
        MapSpecimens(str(tmp_path))


def test_the_endpoint_serves_the_image_with_caching(makeApp, tmp_path):
    writeMap(tmp_path)
    client = makeApp(MAP_SPECIMEN_DIR=str(tmp_path)).test_client()
    response = client.get("/api/map/routes/2.webp")
    assert response.status_code == 200 and response.mimetype == "image/webp"
    assert response.data == b"RIFF-three-longer"
    assert "max-age" in response.headers["Cache-Control"]
    again = client.get("/api/map/routes/2.webp", headers={"If-None-Match": response.headers["ETag"]})
    assert again.status_code == 304


def test_the_endpoint_is_a_404_where_there_is_no_image(makeApp, tmp_path):
    writeMap(tmp_path)
    client = makeApp(MAP_SPECIMEN_DIR=str(tmp_path)).test_client()
    for path in ("/api/map/routes/1.webp", "/api/map/routes/99.webp", "/api/map/flower/0.webp"):
        response = client.get(path)
        assert response.status_code == 404 and response.json == {"message": "Not found"}


def test_a_site_without_the_maps_folder_still_starts(client):
    assert client.get("/api/map/routes/0.webp").status_code == 404
