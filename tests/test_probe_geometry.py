from pyneuroscope.models import ChannelGroup
from pyneuroscope.probe_geometry import (
    available_probe_geometries,
    find_chanmap_file,
    load_chanmap_geometry,
    load_probe_geometry,
    parse_probe_geometry,
    positions_for_pattern,
)
from scipy.io import savemat


def test_parses_slot_geometry_and_maps_group_channels() -> None:
    geometry = parse_probe_geometry(
        """
        {
          "name": "poly2",
          "group_pitch_um": 100,
          "sites": [
            {"slot": 0, "x": -10, "y": 0},
            {"slot": 1, "x": 10, "y": 20}
          ]
        }
        """
    )

    positions = geometry.positions_for_groups(
        [ChannelGroup("a", [5, 6]), ChannelGroup("b", [7, 8])]
    )

    assert positions[5].x == -10
    assert positions[6].y == 20
    assert positions[7].x == 90
    assert positions[8].x == 110


def test_parses_explicit_channel_geometry() -> None:
    geometry = parse_probe_geometry(
        """
        {
          "sites": [
            {"channel": 3, "x": 1.5, "y": 2.5}
          ]
        }
        """,
        fallback_name="custom",
    )

    assert geometry.name == "custom"
    assert geometry.positions_for_groups([ChannelGroup("a", [0])])[3].x == 1.5


def test_loads_geometry_files_by_probe_type(tmp_path) -> None:
    geometry_dir = tmp_path / "probe_geometry"
    geometry_dir.mkdir()
    (geometry_dir / "poly3.json").write_text(
        '{"sites": [{"slot": 0, "x": 0, "y": 0}]}',
        encoding="utf-8",
    )

    names = available_probe_geometries([geometry_dir])
    assert "poly3" in names
    assert "chanMap.mat" not in names
    assert load_probe_geometry("poly3", [geometry_dir]) is not None


def test_builtin_poly3_pattern_uses_channel_order() -> None:
    positions = positions_for_pattern("poly3", [ChannelGroup("shank1", [3, 1, 2, 0])])

    assert positions[3].x == 200
    assert positions[1].x == 182
    assert positions[2].x == 200
    assert positions[0].x == 218
    assert positions[1].y == -20


def test_builtin_probe_pattern_is_loadable_without_json() -> None:
    geometry = load_probe_geometry("poly2", [])

    assert geometry is not None
    positions = geometry.positions_for_groups([ChannelGroup("shank1", [0, 1])])
    assert positions[0].x == 180
    assert positions[1].x == 220


def test_loads_chanmap_mat_geometry(tmp_path) -> None:
    mat_path = tmp_path / "chanMap.mat"
    savemat(
        mat_path,
        {
            "chanMap0ind": [[2, 0, 1]],
            "xcoords": [[20], [0], [10]],
            "ycoords": [[-40], [0], [-20]],
        },
    )

    positions = load_chanmap_geometry(mat_path)

    assert positions[2].x == 20
    assert positions[0].y == 0
    assert positions[1].x == 10


def test_finds_chanmap_file_by_basename_or_default(tmp_path) -> None:
    base = tmp_path / "session"
    base.mkdir()
    named = base / "mouse1.chanMap.mat"
    named.write_bytes(b"placeholder")

    assert find_chanmap_file([base], ["mouse1"]) == named
