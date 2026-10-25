"""frames.py: keyframe detection, video / JPEG / PNG access, materialize."""

import numpy as np
import pytest

import synth
from bebop_vision.lab import frames as fr
from bebop_vision.lab.session import Session


def _au(nal_bytes):
    return b"".join(b"\x00\x00\x00\x01" + n for n in nal_bytes)


def _hevc_nal(nal_type):
    return bytes([(nal_type << 1) & 0x7E, 0x01, 0xAF])


def _h264_nal(nal_type):
    return bytes([0x60 | nal_type, 0x88])


class TestKeyframe:
    def test_hevc_irap_after_parameter_sets(self):
        au = _au([_hevc_nal(32), _hevc_nal(33), _hevc_nal(34), _hevc_nal(39), _hevc_nal(19)])
        assert fr.is_keyframe(au, "h265") is True

    @pytest.mark.parametrize("nal", [16, 17, 18, 19, 20, 21, 22, 23])
    def test_hevc_irap_range(self, nal):
        assert fr.is_keyframe(_au([_hevc_nal(nal)]), "h265") is True

    @pytest.mark.parametrize("nal", [0, 1, 2, 8, 9, 15])
    def test_hevc_non_irap(self, nal):
        assert fr.is_keyframe(_au([_hevc_nal(34), _hevc_nal(nal)]), "hevc") is False

    def test_h264_idr_and_slice(self):
        assert fr.is_keyframe(_au([_h264_nal(7), _h264_nal(8), _h264_nal(5)]), "h264") is True
        assert fr.is_keyframe(_au([_h264_nal(1)]), "h264") is False

    def test_three_byte_start_code(self):
        assert fr.is_keyframe(b"\x00\x00\x01" + _hevc_nal(20), "h265") is True

    def test_jpeg_always_and_no_start_code(self):
        assert fr.is_keyframe(b"\xff\xd8\xff", "jpeg") is True
        assert fr.is_keyframe(b"\x00\x00\x02\x40", "h265") is False
        assert fr.is_keyframe(b"", "h265") is False

    def test_synthetic_stream_keyframes(self):
        aus = synth._encode_video([synth.colour_frame(i) for i in range(20)], "h265")
        flags = [fr.is_keyframe(a, "h265") for a in aus]
        assert flags == [i % 8 == 0 for i in range(20)]


@pytest.fixture(scope="module")
def h265(h265_session, tmp_path_factory):
    return Session(h265_session, cache_dir=tmp_path_factory.mktemp("c") / "h265")


@pytest.fixture(scope="module")
def jpeg(jpeg_session, tmp_path_factory):
    return Session(jpeg_session, cache_dir=tmp_path_factory.mktemp("c") / "jpeg")


class TestFrameSeriesH265:
    def test_index_and_keyframes(self, h265):
        fs = h265.frames("near")
        t = h265.ticks().t_ns
        assert len(fs) == 80 and fs.fmt == "h265"
        np.testing.assert_array_equal(fs.t_ns, t)
        # GOP of 8 in the synthetic encoder
        np.testing.assert_array_equal(fs.keyframes, t[::8])
        assert fs.nearest_t(int(t[3]) + 7_000_000, tol_ns=10_000_000) == int(t[3])
        assert fs.nearest_t(int(t[3]) + 20_000_000, tol_ns=10_000_000) is None

    def test_random_access_marker(self, h265):
        fs = h265.frames("near")
        t = h265.ticks().t_ns
        rng = np.random.default_rng(0)
        for i in rng.permutation(80)[:24]:
            img = fs.at(int(t[i]))
            assert img is not None and img.shape == (80, 128, 3) and img.dtype == np.uint8
            assert synth.tick_of_colour_frame(img) == i % 32, i

    def test_sequential_iter_feeds_each_au_once(self, h265):
        fs = h265.frames("far")
        fs.close()
        fs._lru.clear()
        fs.decode_calls = 0
        t = h265.ticks().t_ns
        got = [synth.tick_of_colour_frame(f) for _, f in fs.iter(t)]
        assert got == [i % 32 for i in range(80)]
        # the decoder stayed open: one AU fed per frame, no GOP re-decodes
        assert fs.decode_calls == 80

    def test_seek_backwards_restarts_at_keyframe(self, h265):
        fs = h265.frames("near")
        fs.close()
        fs._lru.clear()
        t = h265.ticks().t_ns
        assert synth.tick_of_colour_frame(fs.at(int(t[30]))) == 30
        fs._lru.clear()
        fs.decode_calls = 0
        assert synth.tick_of_colour_frame(fs.at(int(t[12]))) == 12
        # keyframe at 8, so exactly 8..12 were fed
        assert fs.decode_calls == 5
        # the tail frame needs a flush at the end of the stream
        assert synth.tick_of_colour_frame(fs.at(int(t[79]))) == 79 % 32

    def test_lru_hit_costs_no_decode(self, h265):
        fs = h265.frames("near")
        t = h265.ticks().t_ns
        fs.at(int(t[40]))
        n = fs.decode_calls
        fs.at(int(t[40]))
        assert fs.decode_calls == n

    def test_lru_bound(self, h265):
        fs = h265.frames("near")
        t = h265.ticks().t_ns
        for ti in t:
            fs.at(int(ti))
        assert len(fs._lru) <= fs.cache_frames

    def test_missing_stamp_is_none(self, h265):
        fs = h265.frames("near")
        assert fs.at(int(h265.ticks().t_ns[0]) + 1) is None
        assert fs.frame_at_index(10_000) is None


def _dropped(i):
    return i > 0 and i % 5 == 0


def _after_hole(i):
    """A far AU in the dropped_far fixture whose GOP (8 AUs) has a hole
    before it: ffmpeg would conceal it against a flat grey reference."""
    return any(_dropped(h) for h in range((i // 8) * 8, i))


class TestDroppedFar:
    def test_dropped_far_frames_are_none(self, dropped_far_session, tmp_path):
        s = Session(dropped_far_session, cache_dir=tmp_path / "drop")
        t = s.ticks().t_ns
        flags = s.flags()
        fs = s.frames("far")
        assert len(fs) == 80 - len([i for i in range(1, 80) if _dropped(i)])
        for i in range(80):
            assert bool(flags["has_color_far"][i]) is (not _dropped(i))
            img = fs.at(int(t[i]))
            if _dropped(i) or _after_hole(i):
                # no AU, or an AU whose reference the stream never carried:
                # None, never a concealed grey picture
                assert img is None, i
            else:
                assert synth.tick_of_colour_frame(img) == i % 32, i
        assert flags["has_color_near"].all()
        # the taint comes from the tick grid, before any decode; the holes at
        # 15, 55 sit right before keyframes 16, 56 and cost nothing
        rows = [i for i in range(80) if not _dropped(i)]
        assert fs.tainted.tolist() == [_after_hole(i) for i in rows]
        assert not fs.tainted[rows.index(16)] and not fs.tainted[rows.index(56)]
        sel = s.ticks().select("color_near", "color_far")
        smp = sel[10]
        assert smp["color_far"] is None and smp["color_near"] is not None
        assert smp.flags["has_color_far"] is False
        # recorded but undecodable: the AU flag stays, the frame is None
        smp = sel[6]
        assert smp["color_far"] is None and smp.flags["has_color_far"] is True
        # the near stream has no hole and is untouched
        assert not s.frames("near").tainted.any()


class TestPairing:
    """Frames must land on their own tick when an AU yields no frame."""

    @pytest.fixture
    def near_index(self, h265):
        h265.index()
        return h265.ticks().t_ns, h265.cache_dir / "video_near.bin", np.load(h265.cache_dir / "video_near.npy")

    def test_damaged_au_taints_rest_of_gop(self, near_index, tmp_path):
        t, bin_path, idx = near_index
        bad = idx.copy()
        bad["size"][10] = 24           # a truncated slice inside GOP 8..15
        np.save(tmp_path / "v.npy", bad)
        fs = fr.FrameSeries(bin_path, tmp_path / "v.npy", "near", ticks=t)
        assert synth.tick_of_colour_frame(fs.at(int(t[9]))) == 9
        assert fs.at(int(t[10])) is None
        for i in (11, 12, 15):
            assert fs.at(int(t[i])) is None, i
        assert fs.tainted[10:16].all() and not fs.tainted[16:].any() and not fs.tainted[:10].any()
        for i in (16, 17, 23):
            assert synth.tick_of_colour_frame(fs.at(int(t[i]))) == i % 32
        # a later request inside the tainted GOP costs no decode
        n = fs.decode_calls
        assert fs.at(int(t[13])) is None and fs.decode_calls == n

    def test_segment_starting_mid_gop(self, near_index, tmp_path):
        t, bin_path, idx = near_index
        np.save(tmp_path / "v.npy", idx[5:].copy())    # first keyframe is row 3 (AU 8)
        fs = fr.FrameSeries(bin_path, tmp_path / "v.npy", "near")
        assert fs.tainted[:3].all() and not fs.tainted[3:].any()
        for row in range(0, 3):
            assert fs.frame_at_index(row) is None
        for row in range(3, 12):
            assert synth.tick_of_colour_frame(fs.frame_at_index(row)) == (row + 5) % 32, row

    def test_pts_pairing_survives_silent_aus(self, near_index, tmp_path):
        """With the taint cleared, the three P-slices before the first IRAP
        raise and emit nothing; pairing by pts still puts frame 8 on row 3,
        where pairing by emission order put it on row 0."""
        t, bin_path, idx = near_index
        np.save(tmp_path / "v.npy", idx[5:].copy())
        fs = fr.FrameSeries(bin_path, tmp_path / "v.npy", "near")
        fs._tainted[:] = False
        got = [fs.frame_at_index(r) for r in range(0, 6)]
        assert got[:3] == [None, None, None]
        assert [synth.tick_of_colour_frame(g) for g in got[3:]] == [8, 9, 10]
        # the failures were learnt while decoding
        assert fs.tainted[:3].all()


class TestJpegAndDepth:
    def test_jpeg_random_access(self, jpeg):
        fs = jpeg.frames("near")
        t = jpeg.ticks().t_ns
        assert fs.fmt == "jpeg" and len(fs.keyframes) == 80
        for i in (0, 17, 63, 79):
            img = fs.at(int(t[i]))
            assert img.shape == (80, 128, 3)
            assert synth.tick_of_colour_frame(img) == i % 32
        assert fs.decode_calls == 0   # no video decoder involved

    def test_depth_pixel_is_tick_index(self, h265, jpeg):
        for s in (h265, jpeg):
            t = s.ticks().t_ns
            for role in ("near", "far"):
                ds = s.depth(role)
                assert len(ds) == 80
                for i in (0, 9, 45, 79):
                    d = ds.at(int(t[i]))
                    assert d.dtype == np.uint16 and d.shape == (60, 106)
                    assert int(d[0, 0]) == i
                    assert int(d[30, 50]) == 1500
                assert ds.at(int(t[0]) - 1) is None

    def test_empty_series(self, tmp_path):
        fs = fr.FrameSeries(tmp_path / "none.bin", tmp_path / "none.npy")
        assert len(fs) == 0 and fs.at(0) is None and fs.nearest_t(0) is None
        assert len(fs.keyframes) == 0
        ds = fr.DepthSeries(tmp_path / "none.bin", tmp_path / "none.npy")
        assert len(ds) == 0 and ds.at(0) is None


class TestMaterialize:
    def test_writes_memmaps(self, h265, tmp_path):
        t = h265.ticks(hz=10).t_ns[:10]
        out = fr.materialize(h265, ("near", "far"), t, size=(64, 40), out_dir=tmp_path / "mat")
        arr = np.load(out / "frames_near.npy", mmap_mode="r")
        assert arr.shape == (10, 40, 64, 3) and arr.dtype == np.uint8
        np.testing.assert_array_equal(np.load(out / "t_ns.npy"), t)
        assert np.load(out / "valid_near.npy").all()
        far = np.load(out / "frames_far.npy", mmap_mode="r")
        assert far.shape == (10, 40, 64, 3)
        # the marker block survives the resize (16x16 -> 8x8 at the top-left)
        grey = float(np.asarray(arr[3])[1:6, 1:6].mean())
        assert round(grey / 8.0) % 32 == 6   # tick index 6 at 10 Hz sample 3

    def test_missing_frames_are_invalid_rows(self, dropped_far_session, tmp_path):
        s = Session(dropped_far_session, cache_dir=tmp_path / "drop")
        t = s.ticks().t_ns[:12]
        out = fr.materialize(s, ("far",), t, out_dir=tmp_path / "mat")
        valid = np.load(out / "valid_far.npy")
        assert valid.tolist() == [not (_dropped(i) or _after_hole(i)) for i in range(12)]
        arr = np.load(out / "frames_far.npy", mmap_mode="r")
        assert arr.shape == (12, 80, 128, 3)
        assert not np.asarray(arr[5]).any() and not np.asarray(arr[6]).any()

    def test_meta_identity_and_keyed_directory(self, h265):
        import json
        from bebop_vision.lab.types import file_identity
        t = h265.ticks().t_ns
        a = fr.materialize(h265, ("near",), t[:4])
        b = fr.materialize(h265, ("far",), t[:6])
        assert a != b and a.parent == b.parent == h265.cache_dir / "materialized"
        meta = json.loads((a / "meta.json").read_text())
        assert meta["identity"] == file_identity(h265.path) and meta["producer"] == fr.PRODUCER
        assert meta["source"] == str(h265.path) and meta["roles"] == ["near"] and meta["n"] == 4
        assert meta["key"] == a.name == fr.materialize_key(("near",), t[:4])
        np.testing.assert_array_equal(np.load(a / "t_ns.npy"), t[:4])
        np.testing.assert_array_equal(np.load(b / "t_ns.npy"), t[:6])
        # same request: reused, not rebuilt; force rebuilds
        built = meta["built_at"]
        (a / "marker").write_text("x")
        assert fr.materialize(h265, ("near",), t[:4]) == a and (a / "marker").exists()
        assert json.loads((a / "meta.json").read_text())["built_at"] == built
        import time
        time.sleep(1.1)
        fr.materialize(h265, ("near",), t[:4], force=True)
        assert json.loads((a / "meta.json").read_text())["built_at"] != built
        # the key separates size and tick lists too
        assert fr.materialize_key(("near",), t[:4], size=(64, 40)) != meta["key"]
        assert fr.materialize_key(("near",), t[:5]) != meta["key"]
