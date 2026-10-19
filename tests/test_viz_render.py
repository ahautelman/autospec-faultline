"""Tests for loading a data directory, choosing figures, and the command line.

Everything runs in temporary directories with hand-built reports, so no training happens
and no real report directory is touched.
"""

from __future__ import annotations


import pytest

pytest.importorskip("matplotlib", reason="figure tests need the reports extra")

from faultline.envs import layout_to_dict  # noqa: E402
from faultline.experiments import rooms9_autospec, rooms9_report  # noqa: E402
from faultline.experiments.autospec.records import AutoSpecReport, report_to_dict  # noqa: E402
from faultline.experiments.records import (  # noqa: E402
    Checkpoint,
    Overlay,
    ScenarioReport,
    ScenarioSpread,
    SeedSummary,
    spread_to_dict,
    write_json,
)
from faultline.experiments.records import report_to_dict as scenario_report_to_dict  # noqa: E402
from faultline.graph.snapshot import snapshot_graph  # noqa: E402
from faultline.refinement.loop import RefinementStep  # noqa: E402
from faultline.scenarios import SCENARIOS  # noqa: E402
from faultline.viz import cli  # noqa: E402
from faultline.viz.load import load_data  # noqa: E402
from faultline.viz.render import RenderOptions, render, render_directory  # noqa: E402

STEPS = (10_000, 80_000, 200_000)


def _curve(*values):
    return tuple(Checkpoint(s, v) for s, v in zip(STEPS, values, strict=True))


def _autospec(scenario="narrow_door", seed=0, *, geometry=True, curve=True) -> AutoSpecReport:
    sc = SCENARIOS[scenario]
    graph = snapshot_graph(sc.build_graph())
    extra = dict(schema_version=2, layout=layout_to_dict(sc.layout), initial_graph=graph,
                 final_graph=graph) if geometry else {}
    return AutoSpecReport(
        scenario=scenario, seed=seed, expected_operator="seq",
        config={"autospec": {"threshold": 0.99}, "training": {"budget": 200_000}},
        steps=(RefinementStep("start", "goal", ("seq",), None, 0.3, 0.3),),
        initial_edges=(), final_edges=(), best_path_before=0.3, best_path_after=0.9,
        reference_before=0.3, reference_after=0.75,
        before_curve=_curve(0.1, 0.2, 0.3) if curve else (Checkpoint(200_000, 0.3),),
        after_curve=_curve(0.5, 0.7, 0.9) if curve else (Checkpoint(200_000, 0.9),),
        elapsed_seconds=1.0, **extra,
    )


def _save(directory, name, payload):
    write_json(directory / name, payload)


def _autospec_dir(tmp_path, *reports):
    data = tmp_path / "data"
    for report in reports:
        _save(data, f"autospec_{report.scenario}_seed{report.seed}.json", report_to_dict(report))
    return data


def _names(result):
    return sorted(path.name for path in result.paths)


def test_full_loop_reports_are_loaded_in_catalogue_order_then_seed_order(tmp_path):
    data = _autospec_dir(tmp_path, _autospec("mixed_start", 1), _autospec("trap_goal", 1),
                         _autospec("trap_goal", 0))

    loaded = load_data(data)

    assert [(r.scenario, r.seed) for r in loaded.runs] == [("trap_goal", 0), ("trap_goal", 1), ("mixed_start", 1)]


def test_files_are_recognised_by_content_not_name(tmp_path):
    data = tmp_path / "data"
    _save(data, "renamed.json", report_to_dict(_autospec()))
    _save(data, "notes.json", {"hello": "world"})

    loaded = load_data(data)

    assert len(loaded.runs) == 1
    assert loaded.notes == ("skipped notes.json: not a run report",)


def test_toy_reports_and_summaries_load_into_runs_and_seed_runs(tmp_path):
    data = tmp_path / "data"
    toy = ScenarioReport("trap_goal", "seq", 0, 200_000, _curve(0.1, 0.2, 0.3), _curve(0.5, 0.7, 0.9),
                         Overlay("edge", (), (), None, None, None), 0.15, 0.85)
    spread = ScenarioSpread("trap_goal", 200_000, (SeedSummary(0, 0.2, 0.7, 0.3, 0.9), SeedSummary(1, 0.25, 0.8, 0.35, 0.95)))
    _save(data, "step1_trap_goal.json", scenario_report_to_dict(toy))
    _save(data, "step2_trap_goal.json", spread_to_dict(spread))

    loaded = load_data(data)

    assert [r.seed for r in loaded.runs] == [0]
    assert [r.seed for r in loaded.seed_runs] == [0, 1]
    assert [c.probability for c in loaded.seed_runs[1].after_curve] == [0.8, 0.95]


def test_toy_report_of_an_unknown_scenario_is_skipped_with_a_note(tmp_path):
    data = tmp_path / "data"
    toy = ScenarioReport("mystery", "seq", 0, 10, _curve(0, 0, 0), _curve(0, 0, 0),
                         Overlay("edge", (), (), None, None, None), 0.1, 0.9)
    _save(data, "step1_mystery.json", scenario_report_to_dict(toy))

    loaded = load_data(data)

    assert loaded.runs == () and "not in the catalogue" in loaded.notes[0]


def test_two_runs_with_one_scenario_and_seed_are_rejected(tmp_path):
    data = tmp_path / "data"
    _save(data, "a.json", report_to_dict(_autospec()))
    _save(data, "b.json", report_to_dict(_autospec()))

    with pytest.raises(ValueError, match="mix different experiments"):
        load_data(data)


def test_missing_directory_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_data(tmp_path / "nowhere")


def test_everything_the_data_supports_is_drawn(tmp_path):
    data = _autospec_dir(tmp_path, _autospec("narrow_door", 0), _autospec("narrow_door", 1))

    result = render(load_data(data), tmp_path / "figs", RenderOptions(formats=("png",)))

    assert _names(result) == sorted([
        "overlay_narrow_door_seed0.png", "overlay_narrow_door_seed0.caption.txt",
        "curves_narrow_door.png", "curves_narrow_door.caption.txt",
        "seeds_narrow_door.png", "seeds_narrow_door.caption.txt",
        "reference_comparison.png", "reference_comparison.caption.txt",
    ])


def test_several_scenarios_also_get_combined_curve_and_seed_figures(tmp_path):
    data = _autospec_dir(tmp_path, *(_autospec(n, s) for n in ("trap_goal", "narrow_door") for s in (0, 1)))

    names = _names(render(load_data(data), tmp_path / "figs", RenderOptions(formats=("png",))))

    assert "curves_all.png" in names and "seeds_all.png" in names


def test_old_report_without_geometry_skips_the_overlay_with_a_note(tmp_path):
    data = _autospec_dir(tmp_path, _autospec(geometry=False))

    result = render(load_data(data), tmp_path / "figs", RenderOptions(figures=["overlay"], formats=("png",)))

    assert result.paths == ()
    assert any("geometry was not recorded" in note for note in result.notes)


def test_run_that_kept_only_the_final_checkpoint_has_no_curves(tmp_path):
    data = _autospec_dir(tmp_path, _autospec(curve=False))

    result = render(load_data(data), tmp_path / "figs", RenderOptions(figures=["curves"], formats=("png",)))

    assert result.paths == () and any("rerun with --curves" in note for note in result.notes)


def test_single_seed_gets_a_note_instead_of_a_seed_figure(tmp_path):
    data = _autospec_dir(tmp_path, _autospec())

    result = render(load_data(data), tmp_path / "figs", RenderOptions(figures=["seeds"], formats=("png",)))

    assert result.paths == () and any("at least 2 seeds" in note for note in result.notes)


def test_comparison_is_skipped_when_no_run_has_paper_numbers(tmp_path):
    data = tmp_path / "data"
    bare = _autospec()
    payload = report_to_dict(bare)
    payload["reference_before"] = payload["reference_after"] = None
    _save(data, "x.json", payload)

    result = render(load_data(data), tmp_path / "figs", RenderOptions(figures=["reference"], formats=("png",)))

    assert result.paths == () and any("paper's reference numbers" in n for n in result.notes)


def test_scenario_and_seed_filters_pick_what_is_drawn(tmp_path):
    data = _autospec_dir(tmp_path, *(_autospec(n, s) for n in ("trap_goal", "narrow_door") for s in (0, 1)))
    options = RenderOptions(figures=["overlay"], scenarios=["narrow_door"], seeds=[0, 1], formats=("png",))

    names = _names(render(load_data(data), tmp_path / "figs", options))

    assert [n for n in names if n.endswith(".png")] == ["overlay_narrow_door_seed0.png", "overlay_narrow_door_seed1.png"]


def test_overlay_defaults_to_the_lowest_seed_per_scenario(tmp_path):
    data = _autospec_dir(tmp_path, _autospec(seed=3), _autospec(seed=5))

    names = _names(render(load_data(data), tmp_path / "figs", RenderOptions(figures=["overlay"], formats=("png",))))

    assert "overlay_narrow_door_seed3.png" in names and "overlay_narrow_door_seed5.png" not in names


def test_unknown_figure_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown figure kind"):
        render(load_data(_autospec_dir(tmp_path, _autospec())), tmp_path / "f", RenderOptions(figures=["pie"]))


def test_toy_data_draws_curves_comparison_and_per_seed_figures(tmp_path):
    data = tmp_path / "data"
    toy = ScenarioReport("trap_goal", "seq", 0, 200_000, _curve(0.1, 0.2, 0.3), _curve(0.5, 0.7, 0.9),
                         Overlay("edge", (), (), None, None, None), 0.15, 0.85)
    spread = ScenarioSpread("trap_goal", 200_000, (SeedSummary(0, 0.2, 0.7, 0.3, 0.9), SeedSummary(1, 0.25, 0.8, 0.35, 0.95)))
    _save(data, "step1_trap_goal.json", scenario_report_to_dict(toy))
    _save(data, "step2_trap_goal.json", spread_to_dict(spread))

    names = _names(render(load_data(data), tmp_path / "figs", RenderOptions(formats=("png",))))

    assert {"curves_trap_goal.png", "seeds_trap_goal.png", "reference_comparison.png"} <= set(names)
    assert "overlay_trap_goal_seed0.png" not in names


def test_render_directory_prints_notes_and_a_summary(tmp_path, capsys):
    data = _autospec_dir(tmp_path, _autospec(geometry=False))

    render_directory(data, tmp_path / "figs", RenderOptions(figures=["overlay"], formats=("png",)))

    out = capsys.readouterr().out
    assert "note: overlay for narrow_door seed 0" in out and "Wrote 0 figure files" in out


def test_command_line_defaults_the_output_next_to_the_data(tmp_path):
    data = _autospec_dir(tmp_path, _autospec())

    cli.main([str(data), "--figures", "curves", "--formats", "png"])

    assert (tmp_path / "figures" / "curves_narrow_door.png").exists()


def test_command_line_filters_and_embeds_captions(tmp_path):
    data = _autospec_dir(tmp_path, _autospec("trap_goal"), _autospec("narrow_door"))

    cli.main([str(data), "--output", str(tmp_path / "out"), "--scenario", "trap_goal", "--figures", "overlay",
              "--per-step", "--embed-caption", "--formats", "png"])

    assert sorted(p.name for p in (tmp_path / "out").glob("*.png")) == ["overlay_trap_goal_seed0.png"]


def test_autospec_plot_only_redraws_into_a_figures_folder(tmp_path):
    _autospec_dir(tmp_path, _autospec("narrow_door", 0), _autospec("narrow_door", 1))

    rooms9_autospec.main(["--plot-only", "--output", str(tmp_path), "--scenarios", "narrow_door", "--seeds", "0", "1"])

    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "figures" / "curves_narrow_door.png").exists()


def test_toy_report_plot_only_redraws_into_a_figures_folder(tmp_path):
    data = tmp_path / "data"
    toy = ScenarioReport("trap_goal", "seq", 0, 200_000, _curve(0.1, 0.2, 0.3), _curve(0.5, 0.7, 0.9),
                         Overlay("edge", (), (), None, None, None), 0.15, 0.85)
    _save(data, "step1_trap_goal.json", scenario_report_to_dict(toy))

    rooms9_report.main(["--plot-only", "--output", str(tmp_path)])

    assert (tmp_path / "figures" / "curves_trap_goal.png").exists()
