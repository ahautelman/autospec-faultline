from faultline.faults import all_fault_cases, missing_alternative_case


def _has_edge(graph, source: str, target: str) -> bool:
    return any(
        edge.source.name == source and edge.target.name == target
        for edge in graph.edges
    )


def test_all_fault_cases_are_valid_independent_dags():
    cases = all_fault_cases()
    assert len(cases) == 5

    for benchmark in cases:
        case = benchmark.fault
        case.correct_graph.assert_acyclic()
        case.faulty_graph.assert_acyclic()
        assert case.correct_graph is not case.faulty_graph


def test_expected_operators_cover_all_four_autospec_operators():
    expected = {
        benchmark.fault.expected_operator
        for benchmark in all_fault_cases()
    }
    assert expected == {"seq", "past", "add", "or"}


def test_missing_alternative_has_no_unrelated_start_to_b_bypass():
    case = missing_alternative_case().fault

    # Ground truth needs the A->B bridge.
    assert _has_edge(case.correct_graph, "a", "b")

    # Fault injection removes exactly that bridge.
    assert not _has_edge(case.faulty_graph, "a", "b")

    # There must not be an independent start->B route, otherwise the whole
    # specification can already succeed without the missing alternative.
    assert not _has_edge(case.correct_graph, "start", "b")
    assert not _has_edge(case.faulty_graph, "start", "b")

    # B remains an alternative parent of goal, which is what OrRefine needs.
    assert _has_edge(case.faulty_graph, "b", "goal")
    assert _has_edge(case.faulty_graph, "a", "goal")
