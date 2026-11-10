import json

import numpy as np
import pytest
import soundfile as sf
from conftest import TOY_LENGTH, TOY_RATE, make_tone

from esc_robustness.cli import main
from esc_robustness.data import load_metadata
from esc_robustness.dsp import FeatureConfig
from esc_robustness.mixing import (
    DemandCache,
    InterferenceConfig,
    MixtureRenderer,
    active_mask,
    active_power,
    activity_overlap,
    build_manifest,
    interferer_gain,
    manifest_digest,
    masked_power,
    mix,
    pair_distractors,
    prepare_demand,
    read_manifest,
    render_examples,
)

RATE, LENGTH = TOY_RATE, TOY_LENGTH


def _tone(frequency, length=LENGTH, amplitude=0.3):
    return make_tone(frequency, length, amplitude)


@pytest.fixture
def demand_cache(tmp_path):
    demand_root = tmp_path / "DEMAND"
    rng = np.random.default_rng(3)
    for environment in ("STRAFFIC", "PSTATION"):
        (demand_root / environment).mkdir(parents=True)
        noise = 0.05 * rng.standard_normal(3 * 48000)
        sf.write(demand_root / environment / "ch01.wav", noise, 48000, subtype="PCM_16")
    cache = tmp_path / "demand_cache"
    prepare_demand(demand_root, cache, ("STRAFFIC", "PSTATION"))
    return cache


def test_active_mask_follows_frame_energy():
    config = FeatureConfig()
    x = _tone(440)
    x[: LENGTH // 2] = 0.0
    mask = active_mask(x, config)
    assert not mask[: LENGTH // 2 - config.frame_length].any()
    assert mask[LENGTH // 2 + config.frame_length :].all()
    assert active_mask(np.zeros(4096), config).all()
    with pytest.raises(ValueError):
        masked_power(x, np.zeros(LENGTH, dtype=bool))


def test_gain_reaches_requested_ratio_and_common_gain_preserves_it():
    target, interferer = _tone(440, amplitude=0.9), _tone(1234, amplitude=0.7)
    p_target, mask = active_power(target)
    p_interferer, interferer_mask = active_power(interferer)
    assert p_target == pytest.approx(0.9**2 / 2, rel=1e-3)
    for level in (10.0, 5.0, 0.0, -3.0):
        gain = interferer_gain(p_target, p_interferer, level)
        mixture, common, peak = mix(target, interferer, gain, peak_limit=0.99)
        assert np.max(np.abs(mixture)) <= 0.99 + 1e-12
        assert peak >= np.max(np.abs(mixture))
        ratio = 10 * np.log10(
            masked_power(target * common, mask)
            / masked_power(gain * common * interferer, interferer_mask)
        )
        assert ratio == pytest.approx(level, abs=1e-9)
    # Zero-level mixture of two 0.9/0.7 tones must exceed the peak limit.
    assert mix(target, interferer, interferer_gain(p_target, p_interferer, 0.0))[1] < 1.0
    quiet = mix(0.1 * target, interferer, interferer_gain(0.01 * p_target, p_interferer, 10.0))
    assert quiet[1] == 1.0
    with pytest.raises(ValueError):
        interferer_gain(p_target, 0.0, 0.0)


def test_sparse_event_is_measured_only_where_it_sounds():
    sparse = _tone(600, amplitude=0.5)
    sparse[LENGTH // 4 :] = 0.0  # one short burst, then silence
    dense = _tone(600, amplitude=0.5)
    p_sparse, mask = active_power(sparse)
    p_dense, _ = active_power(dense)
    assert mask.mean() == pytest.approx(0.25, abs=0.05)
    assert p_sparse == pytest.approx(p_dense, rel=0.15)  # boundary frame adds a little silence
    assert masked_power(sparse, np.ones(LENGTH, dtype=bool)) < 0.3 * p_dense


def test_activity_overlap():
    first = np.zeros(LENGTH, dtype=bool)
    first[: LENGTH // 2] = True
    assert activity_overlap(first, ~first) == 0.0
    assert activity_overlap(first, first) == 1.0
    assert activity_overlap(first, np.ones(LENGTH, dtype=bool)) == 1.0
    with pytest.raises(ValueError):
        activity_overlap(np.zeros(LENGTH, dtype=bool), first)


def test_pairing_rules_and_determinism(interference_dataset):
    clips = load_metadata(interference_dataset, require_complete=False)
    by_name = {c.filename: c for c in clips}
    config = InterferenceConfig(conditions=("semantic_same", "semantic_different"))
    pairs = pair_distractors(clips, config)
    assert set(pairs) == set(by_name)
    for name, chosen in pairs.items():
        target = by_name[name]
        same, different = chosen["semantic_same"], chosen["semantic_different"]
        for distractor in (same, different):
            assert distractor.fold == target.fold
            assert distractor.source_id != target.source_id
            assert distractor.target != target.target
        assert same.coarse_category == target.coarse_category
        assert different.coarse_category != target.coarse_category
    assert pair_distractors(clips, config) == pairs
    other = pair_distractors(clips, InterferenceConfig(seed=1, conditions=config.conditions))
    assert any(other[n] != pairs[n] for n in pairs)


def test_manifest_round_trip_and_regeneration(interference_dataset, tmp_path):
    config = InterferenceConfig(
        conditions=("white", "semantic_same", "semantic_different"), levels_db=(10.0, 0.0)
    )
    output = tmp_path / "manifest"
    meta = build_manifest(
        interference_dataset, output, config, require_complete=False, limit_per_fold=2
    )
    assert meta["summary"]["rows"] == 5 * 2 * 3 * 2
    read_meta, rows = read_manifest(output)
    assert read_meta["summary"] == meta["summary"]
    assert len(rows) == meta["summary"]["rows"]
    # Rebuilding gives a byte-identical manifest.
    again = tmp_path / "manifest_again"
    build_manifest(interference_dataset, again, config, require_complete=False, limit_per_fold=2)
    assert manifest_digest(output) == manifest_digest(again)
    renderer = MixtureRenderer(interference_dataset, FeatureConfig(), config)
    low_overlap = 0
    for row in rows:
        rendered = renderer.render(row)
        measured = rendered["measurements"]
        for key, value in measured.items():
            assert value == pytest.approx(row[key], rel=1e-9, abs=1e-12), key
        assert row["achieved_db"] == pytest.approx(row["level_db"], abs=1e-6)
        assert row["offset_samples"] == 0
        assert np.max(np.abs(rendered["mixture"])) <= config.peak_limit + 1e-12
        assert rendered["mixture"].shape == (LENGTH,)
        assert rendered["clean_matched"] == pytest.approx(
            renderer.clip(row["target_file"]) * row["common_gain"]
        )
        if row["condition"] == "white":
            assert row["interferer"] == "white" and row["interferer_class"] is None
            assert row["interferer_active_fraction"] == pytest.approx(1.0, abs=0.01)
            assert row["activity_overlap"] == pytest.approx(1.0, abs=0.01)
        else:
            assert row["ratio_type"] == "SIR" and row["interferer_class"] is not None
            same = row["interferer_coarse"] == row["target_coarse"]
            assert same == (row["condition"] == "semantic_same")
            low_overlap += row["activity_overlap"] < 0.5
            # Classes 10 and 11 sound in disjoint halves: the power inside the target's
            # own region is then far below the distractor's active power.
            if row["activity_overlap"] < 0.1:
                assert row["interferer_power_on_target_mask"] < 0.1 * row["interferer_power"]
    assert low_overlap > 0 and meta["summary"]["low_overlap_pairs"] == low_overlap
    # A tampered gain is detected when regenerating.
    bad = dict(rows[0], interferer_gain=rows[0]["interferer_gain"] * 1.01)
    with pytest.raises(ValueError):
        renderer.render(bad)
    # Rows for the same target are identical regardless of the subset size.
    full = tmp_path / "manifest_full"
    build_manifest(interference_dataset, full, config, require_complete=False)
    _, full_rows = read_manifest(full)
    index = {(r["target_file"], r["condition"], r["level_db"]): r for r in full_rows}
    for row in rows:
        twin = index[(row["target_file"], row["condition"], row["level_db"])]
        assert twin["interferer"] == row["interferer"]
        assert twin["interferer_gain"] == pytest.approx(row["interferer_gain"], rel=1e-12)


def test_white_noise_depends_only_on_row_seed(interference_dataset, tmp_path):
    config = InterferenceConfig(conditions=("white",), levels_db=(5.0,))
    build_manifest(interference_dataset, tmp_path / "m", config, require_complete=False)
    _, rows = read_manifest(tmp_path / "m")
    renderer = MixtureRenderer(interference_dataset, FeatureConfig(), config)
    first = renderer.render(rows[0])["mixture"]
    second = MixtureRenderer(interference_dataset, FeatureConfig(), config).render(rows[0])
    assert np.array_equal(first, second["mixture"])
    assert len({r["row_seed"] for r in rows}) == len(rows)
    # The seed depends on (target, condition) only: the same noise realisation serves
    # every level, and neither the row index nor the level set changes it.
    config_two = InterferenceConfig(conditions=("white",), levels_db=(10.0, 5.0))
    build_manifest(interference_dataset, tmp_path / "m2", config_two, require_complete=False)
    _, rows_two = read_manifest(tmp_path / "m2")
    seeds_two = {(r["target_file"], r["level_db"]): r["row_seed"] for r in rows_two}
    for row in rows:
        assert seeds_two[(row["target_file"], 5.0)] == row["row_seed"]
        assert seeds_two[(row["target_file"], 10.0)] == row["row_seed"]
    renderer_two = MixtureRenderer(interference_dataset, FeatureConfig(), config_two)
    by_level = {r["level_db"]: r for r in rows_two if r["target_file"] == rows[0]["target_file"]}
    loud = renderer_two.render(by_level[10.0])
    quiet = renderer_two.render(by_level[5.0])
    # Identical realisation: the scaled interferers differ only by their gains.
    loud_m, quiet_m = loud["measurements"], quiet["measurements"]
    unit_loud = loud["interferer_scaled"] / (loud_m["interferer_gain"] * loud_m["common_gain"])
    unit_quiet = quiet["interferer_scaled"] / (quiet_m["interferer_gain"] * quiet_m["common_gain"])
    assert np.allclose(unit_loud, unit_quiet)


def test_demand_cache_and_manifest(interference_dataset, demand_cache, tmp_path):
    cache = DemandCache(demand_cache)
    assert cache.environments() == ["PSTATION", "STRAFFIC"]
    assert cache.audio("STRAFFIC").size == 3 * RATE
    meta = json.loads((demand_cache / "demand_cache.json").read_text())
    assert meta["sample_rate"] == RATE and meta["STRAFFIC"]["source_sample_rate"] == 48000
    config = InterferenceConfig(
        conditions=("demand",), levels_db=(5.0,), demand_environments=("STRAFFIC", "PSTATION")
    )
    output = tmp_path / "demand_manifest"
    build_manifest(
        interference_dataset, output, config, demand_cache_dir=demand_cache, require_complete=False
    )
    _, rows = read_manifest(output)
    environments = {r["interferer"] for r in rows}
    assert environments == {"demand:STRAFFIC", "demand:PSTATION"}
    renderer = MixtureRenderer(interference_dataset, FeatureConfig(), config, cache)
    for row in rows:
        assert 0 <= row["offset_samples"] <= 3 * RATE - LENGTH
        rendered = renderer.render(row)
        assert rendered["measurements"]["achieved_db"] == pytest.approx(5.0, abs=1e-6)
    with pytest.raises(ValueError):
        build_manifest(interference_dataset, tmp_path / "x", config, require_complete=False)


def test_cli_builds_manifest_and_renders_examples(interference_dataset, tmp_path, capsys):
    config_path = tmp_path / "interference.json"
    config_path.write_text(
        json.dumps(
            {
                "interference": {
                    "conditions": ["white", "semantic_different"],
                    "levels_db": [10.0],
                    "seed": 7,
                }
            }
        )
    )
    manifest = tmp_path / "cli_manifest"
    main(
        [
            "build-manifest",
            "--data-root",
            str(interference_dataset),
            "--output",
            str(manifest),
            "--config",
            str(config_path),
            "--limit-per-fold",
            "1",
            "--allow-incomplete",
        ]
    )
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{") :])
    assert summary["rows"] == 5 * 2
    examples = tmp_path / "examples"
    main(
        [
            "render-mixtures",
            "--manifest",
            str(manifest),
            "--output",
            str(examples),
            "--per-condition",
            "1",
        ]
    )
    written = json.loads((examples / "examples.json").read_text())["files"]
    assert len(written) == 2
    for entry in written:
        audio, rate = sf.read(examples / entry["file"])
        assert rate == RATE and audio.shape == (LENGTH,)
    chosen = render_examples(manifest, tmp_path / "rows", rows=[0, 3])
    assert [c["row"] for c in chosen] == [0, 3]


def test_target_region_power_follows_the_proposal_wording(interference_dataset, tmp_path):
    config = InterferenceConfig(
        conditions=("semantic_same", "semantic_different"),
        levels_db=(5.0,),
        power_region="target",
    )
    output = tmp_path / "target_region"
    meta = build_manifest(interference_dataset, output, config, require_complete=False)
    _, rows = read_manifest(output)
    # Classes 10 and 11 sound in disjoint halves with exact zeros elsewhere, so some
    # same-category pairs have no distractor energy inside the target's region.
    assert meta["summary"]["dropped_silent_in_target_region"] > 0
    assert len(meta["dropped_pairs"]) == meta["summary"]["dropped_silent_in_target_region"]
    assert [r["row"] for r in rows] == list(range(len(rows)))
    assert len(rows) == meta["summary"]["rows"]
    renderer = MixtureRenderer(interference_dataset, FeatureConfig(), config)
    for row in rows:
        target_mask_ratio = 10 * np.log10(
            row["target_power"] / row["interferer_power_on_target_mask"]
        )
        assert target_mask_ratio == pytest.approx(5.0, abs=1e-6)
        assert row["achieved_db"] == pytest.approx(5.0, abs=1e-6)
        rendered = renderer.render(row)
        assert rendered["measurements"]["interferer_gain"] == pytest.approx(
            row["interferer_gain"], rel=1e-9
        )
    # The same pairs under the default definition: interferer_power is the own-region
    # reading, so the two manifests differ only in the gain.
    own = tmp_path / "own_region"
    build_manifest(
        interference_dataset,
        own,
        InterferenceConfig(conditions=config.conditions, levels_db=(5.0,)),
        require_complete=False,
    )
    _, own_rows = read_manifest(own)
    own_index = {(r["target_file"], r["condition"]): r for r in own_rows}
    for row in rows:
        twin = own_index[(row["target_file"], row["condition"])]
        assert twin["interferer"] == row["interferer"]
        assert 10 * np.log10(twin["target_power"] / twin["interferer_power"]) == pytest.approx(
            5.0, abs=1e-6
        )
    with pytest.raises(ValueError):
        InterferenceConfig(power_region="union")
