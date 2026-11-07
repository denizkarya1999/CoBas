from cobas_inference.cli import build_parser, select_chirps


def test_select_chirps_stride_then_limit():
    chirps = list(range(200))
    assert select_chirps(chirps) == chirps
    assert select_chirps(chirps, stride=8) == list(range(0, 200, 8))
    assert select_chirps(chirps, stride=8, limit=25) == list(range(0, 200, 8))[:25]
    assert select_chirps(chirps, limit=10) == list(range(10))
    assert select_chirps(chirps, stride=0, limit=None) == chirps


def test_replay_parser_defaults():
    args = build_parser().parse_args(["replay", "--model", "m.pth", "some/dir"])
    assert args.stride == 1 and args.limit is None and args.csv is None
    args = build_parser().parse_args(["replay", "--model", "m.pth", "some/dir", "--stride", "8", "--limit", "25"])
    assert (args.stride, args.limit) == (8, 25)


def test_predict_parser_accepts_weight_override():
    args = build_parser().parse_args(["predict", "--model", "m.pth", "--weights", "1", "1", "2"])
    assert args.weights == [1.0, 1.0, 2.0]
