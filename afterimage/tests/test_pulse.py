#!/usr/bin/env python3
"""Focused tests for the bounded PULSE 0.1 runtime."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference" / "python"))

import pulse  # noqa: E402


SOURCE = json.loads((ROOT / "content/vertical_slice/cases/PULSE.001.json").read_text(encoding="utf-8"))
CONTRACT = SOURCE["case"]["pulse_contract"]
PROGRAM = SOURCE["case"]["author_program"]
DEDUPLICATE_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.002.json").read_text(encoding="utf-8"))
DEDUPLICATE_CONTRACT = DEDUPLICATE_SOURCE["case"]["pulse_contract"]
DEDUPLICATE_PROGRAM = DEDUPLICATE_SOURCE["case"]["author_program"]
TIMEOUT_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.003.json").read_text(encoding="utf-8"))
TIMEOUT_CONTRACT = TIMEOUT_SOURCE["case"]["pulse_contract"]
TIMEOUT_PROGRAM = TIMEOUT_SOURCE["case"]["author_program"]
TOKEN_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.004.json").read_text(encoding="utf-8"))
TOKEN_CONTRACT = TOKEN_SOURCE["case"]["pulse_contract"]
TOKEN_PROGRAM = TOKEN_SOURCE["case"]["author_program"]
QUORUM_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.005.json").read_text(encoding="utf-8"))
QUORUM_CONTRACT = QUORUM_SOURCE["case"]["pulse_contract"]
QUORUM_PROGRAM = QUORUM_SOURCE["case"]["author_program"]
BARRIER_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.006.json").read_text(encoding="utf-8"))
BARRIER_CONTRACT = BARRIER_SOURCE["case"]["pulse_contract"]
BARRIER_PROGRAM = BARRIER_SOURCE["case"]["author_program"]
BACKPRESSURE_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.007.json").read_text(encoding="utf-8"))
BACKPRESSURE_CONTRACT = BACKPRESSURE_SOURCE["case"]["pulse_contract"]
BACKPRESSURE_PROGRAM = BACKPRESSURE_SOURCE["case"]["author_program"]
FAILOVER_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.008.json").read_text(encoding="utf-8"))
FAILOVER_CONTRACT = FAILOVER_SOURCE["case"]["pulse_contract"]
FAILOVER_PROGRAM = FAILOVER_SOURCE["case"]["author_program"]
EXACTLY_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.009.json").read_text(encoding="utf-8"))
EXACTLY_CONTRACT = EXACTLY_SOURCE["case"]["pulse_contract"]
EXACTLY_PROGRAM = EXACTLY_SOURCE["case"]["author_program"]
DEADLINE_SOURCE = json.loads((ROOT / "content/production/cases/PULSE.010.json").read_text(encoding="utf-8"))
DEADLINE_CONTRACT = DEADLINE_SOURCE["case"]["pulse_contract"]
DEADLINE_PROGRAM = DEADLINE_SOURCE["case"]["author_program"]


class PulseTests(unittest.TestCase):
    def test_standalone_help_is_discoverable(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "pulse.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Player submissions", completed.stdout)
        self.assertIn("player.py", completed.stdout)

    def test_exhaustive_domain_and_boundary_semantics(self) -> None:
        domain = pulse.domain(CONTRACT)
        self.assertEqual(len(domain), 3304)
        self.assertIn((0, 0), domain)
        self.assertIn((11, 11), domain)
        self.assertEqual(pulse.expected_times((0, 5), 5), [10])
        self.assertEqual(pulse.expected_times((0, 6), 5), [5, 11])
        self.assertEqual(pulse.expected_times((0, 0), 5), [5])

    def test_author_program_replaces_timer_and_passes_every_case(self) -> None:
        metrics = pulse.verify_program(PROGRAM, CONTRACT)
        self.assertEqual(metrics, {"program_bytes": 485, "worst_latency": 5, "live_state_cells": 1, "domain_cases": 3304})
        compiled = pulse.compile_program(PROGRAM, pulse.validate_contract(CONTRACT))
        self.assertEqual([item["at"] for item in pulse.run(compiled, CONTRACT, (0, 5)).outputs], [10])
        self.assertEqual([item["at"] for item in pulse.run(compiled, CONTRACT, (0, 6)).outputs], [5, 11])

    def test_static_types_and_resource_failures_are_stable(self) -> None:
        wrong_type = copy.deepcopy(PROGRAM)
        wrong_type["cells"][0]["initial"] = False
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(wrong_type, CONTRACT)
        self.assertEqual(raised.exception.code, "invalid_pulse_program")

        loop = copy.deepcopy(PROGRAM)
        loop["handlers"][1]["actions"] = [{"op": "schedule", "key": "stable-bell", "topic": "pulse.timer", "at": ["event", "at"], "payload": ["event", "payload"]}]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(loop, CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertEqual(raised.exception.context["input"], [0])
        self.assertEqual(raised.exception.context["inner_code"], "pulse_limit")

    def test_deduplicate_ticks_preserves_canonical_first_and_checks_duplicates(self) -> None:
        domain = pulse.domain(DEDUPLICATE_CONTRACT)
        self.assertEqual(len(domain), 1287)
        self.assertIn((0, 0, 0, 0, 0), domain)
        expected = pulse.expected_outputs(DEDUPLICATE_CONTRACT, (1, 1, 3, 3, 3))
        ordered = pulse.input_events(DEDUPLICATE_CONTRACT, (1, 1, 3, 3, 3))
        self.assertEqual(expected[0]["payload"], ordered[0]["payload"])
        self.assertEqual(expected[1]["payload"], next(event["payload"] for event in ordered if event["at"] == 3))
        metrics = pulse.verify_program(DEDUPLICATE_PROGRAM, DEDUPLICATE_CONTRACT)
        self.assertEqual(metrics["worst_latency"], 0)
        self.assertEqual(metrics["live_state_cells"], 1)
        self.assertEqual(metrics["domain_cases"], 1287)

    def test_deduplicate_ticks_returns_smallest_duplicate_counterexample(self) -> None:
        emits_everything = copy.deepcopy(DEDUPLICATE_PROGRAM)
        emits_everything["handlers"][0]["actions"] = [
            {"op": "emit", "topic": "dispatch.unique-ticket", "payload": ["event", "payload"]}
        ]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(emits_everything, DEDUPLICATE_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertEqual(raised.exception.context["input"], [0, 0])

    def test_cancelable_timeout_exhausts_commands_and_deadline_boundary(self) -> None:
        domain = pulse.domain(TIMEOUT_CONTRACT)
        self.assertEqual(len(domain), 545)
        self.assertIn(((0, "start"), (3, "cancel")), domain)
        self.assertEqual(pulse.expected_outputs(TIMEOUT_CONTRACT, ((0, "start"),)), [
            {"at": 3, "topic": "watch.timed-out", "payload": {"watch": "W-3"}},
        ])
        self.assertEqual(pulse.expected_outputs(TIMEOUT_CONTRACT, ((0, "start"), (2, "cancel"))), [])
        self.assertEqual(pulse.expected_outputs(TIMEOUT_CONTRACT, ((0, "start"), (3, "cancel"))), [])
        self.assertEqual(pulse.expected_outputs(TIMEOUT_CONTRACT, ((0, "start"), (2, "start"))), [
            {"at": 5, "topic": "watch.timed-out", "payload": {"watch": "W-3"}},
        ])
        metrics = pulse.verify_program(TIMEOUT_PROGRAM, TIMEOUT_CONTRACT)
        self.assertEqual(metrics["worst_latency"], 3)
        self.assertEqual(metrics["live_state_cells"], 0)
        self.assertEqual(metrics["domain_cases"], 545)

    def test_cancelable_timeout_requires_real_cancel_and_stable_counterexample(self) -> None:
        ignores_cancel = copy.deepcopy(TIMEOUT_PROGRAM)
        ignores_cancel["handlers"] = [handler for handler in ignores_cancel["handlers"] if handler["on"] != "watch.cancel"]
        contexts = []
        for _attempt in range(2):
            with self.assertRaises(pulse.PulseError) as raised:
                pulse.verify_program(ignores_cancel, TIMEOUT_CONTRACT)
            self.assertEqual(raised.exception.code, "pulse_counterexample")
            contexts.append(raised.exception.context)
        self.assertEqual(contexts[0], contexts[1])
        self.assertTrue(any(command["kind"] == "cancel" for command in contexts[0]["input"]))

        malformed = copy.deepcopy(TIMEOUT_PROGRAM)
        malformed["handlers"][1]["actions"][0]["key"] = ""
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(malformed, TIMEOUT_CONTRACT)
        self.assertEqual(raised.exception.code, "invalid_pulse_program")

    def test_token_bucket_exhausts_bursts_refill_and_rejection_boundary(self) -> None:
        exhaustive = pulse.domain(TOKEN_CONTRACT)
        self.assertEqual(len(exhaustive), 924)
        self.assertIn((0, 0, 0, 1), exhaustive)
        self.assertIn((0, 0, 3, 3, 3), exhaustive)
        expected = pulse.expected_outputs(TOKEN_CONTRACT, (0, 0, 0, 1))
        self.assertEqual([item["at"] for item in expected], [0, 0, 1])
        self.assertEqual(len({item["payload"]["sequence"] for item in expected}), 3)
        compiled = pulse.compile_program(TOKEN_PROGRAM, pulse.validate_contract(TOKEN_CONTRACT))
        self.assertEqual(pulse.run(compiled, TOKEN_CONTRACT, (0, 0, 0, 1)).outputs, expected)
        metrics = pulse.verify_program(TOKEN_PROGRAM, TOKEN_CONTRACT)
        self.assertEqual(metrics["worst_latency"], 0)
        self.assertEqual(metrics["live_state_cells"], 2)
        self.assertEqual(metrics["domain_cases"], 924)

    def test_token_bucket_rejects_over_admission_and_unclamped_credit(self) -> None:
        over_admits = copy.deepcopy(TOKEN_PROGRAM)
        over_admits["handlers"][0]["actions"][2]["condition"] = ["const", True]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(over_admits, TOKEN_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertEqual(raised.exception.context["input"], [0, 0, 0])

        no_clamp = copy.deepcopy(TOKEN_PROGRAM)
        no_clamp["handlers"][0]["actions"][0]["value"] = no_clamp["handlers"][0]["actions"][0]["value"][2]
        contexts = []
        for _attempt in range(2):
            with self.assertRaises(pulse.PulseError) as raised:
                pulse.verify_program(no_clamp, TOKEN_CONTRACT)
            self.assertEqual(raised.exception.code, "pulse_counterexample")
            contexts.append(raised.exception.context)
        self.assertEqual(contexts[0], contexts[1])
        self.assertGreater(len(contexts[0]["observed"]), len(contexts[0]["expected"]))

    def test_token_bucket_rejected_request_does_not_consume_credit(self) -> None:
        charges_rejected = copy.deepcopy(TOKEN_PROGRAM)
        admitted = charges_rejected["handlers"][0]["actions"][2]
        decrement = admitted["actions"].pop()
        charges_rejected["handlers"][0]["actions"].append(decrement)
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(charges_rejected, TOKEN_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        counterexample = raised.exception.context["input"]
        self.assertGreaterEqual(max(counterexample), 1)
        self.assertLess(len(set(counterexample)), len(counterexample))

        malformed = copy.deepcopy(TOKEN_PROGRAM)
        malformed["handlers"][0]["actions"][0]["value"][1] = ["const", True]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(malformed, TOKEN_CONTRACT)
        self.assertEqual(raised.exception.code, "invalid_pulse_program")

    def test_backpressure_exhausts_capacity_drain_and_empty_drain(self) -> None:
        self.assertEqual(pulse.verify_program(BACKPRESSURE_PROGRAM, BACKPRESSURE_CONTRACT), {
            "program_bytes": 575, "worst_latency": 0,
            "live_state_cells": 1, "domain_cases": 2561,
        })
        commands = ((0, "request"), (1, "request"), (2, "drain"), (3, "request"))
        expected = pulse.expected_outputs(BACKPRESSURE_CONTRACT, commands)
        self.assertEqual([item["payload"]["sequence"] for item in expected], [0, 1, 3])
        compiled = pulse.compile_program(BACKPRESSURE_PROGRAM, pulse.validate_contract(BACKPRESSURE_CONTRACT))
        self.assertEqual(pulse.run(compiled, BACKPRESSURE_CONTRACT, commands).outputs, expected)

    def test_backpressure_rejects_over_admission_and_ignored_drain(self) -> None:
        over_admits = copy.deepcopy(BACKPRESSURE_PROGRAM)
        over_admits["handlers"][0]["actions"][0]["condition"] = ["const", True]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(over_admits, BACKPRESSURE_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertGreaterEqual(sum(item["source"] == "request" for item in raised.exception.context["input"]), 3)

        ignores_drain = copy.deepcopy(BACKPRESSURE_PROGRAM)
        ignores_drain["handlers"][1]["actions"] = []
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(ignores_drain, BACKPRESSURE_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertTrue(any(item["source"] == "drain" for item in raised.exception.context["input"]))

    def test_backpressure_rejection_is_not_a_hidden_debit(self) -> None:
        charges_rejected = copy.deepcopy(BACKPRESSURE_PROGRAM)
        guarded = charges_rejected["handlers"][0]["actions"][0]
        increment = guarded["actions"].pop()
        charges_rejected["handlers"][0]["actions"].append(increment)
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(charges_rejected, BACKPRESSURE_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        sources = [item["source"] for item in raised.exception.context["input"]]
        self.assertIn("drain", sources)
        self.assertGreaterEqual(sources.count("request"), 3)

    def test_warm_failover_exhausts_failure_readiness_and_recovery(self) -> None:
        self.assertEqual(pulse.verify_program(FAILOVER_PROGRAM, FAILOVER_CONTRACT), {
            "program_bytes": 928, "worst_latency": 0,
            "live_state_cells": 2, "domain_cases": 10417,
        })
        commands = ((0, "warm"), (1, "fail"), (2, "request"), (3, "recover"), (4, "request"))
        expected = pulse.expected_outputs(FAILOVER_CONTRACT, commands)
        self.assertEqual([item["topic"] for item in expected], ["dispatch.to-secondary", "dispatch.to-primary"])
        compiled = pulse.compile_program(FAILOVER_PROGRAM, pulse.validate_contract(FAILOVER_CONTRACT))
        self.assertEqual(pulse.run(compiled, FAILOVER_CONTRACT, commands).outputs, expected)

    def test_warm_failover_rejects_cold_takeover_and_ignored_recovery(self) -> None:
        cold_takeover = copy.deepcopy(FAILOVER_PROGRAM)
        cold_takeover["handlers"][2]["actions"][1]["condition"] = ["not", ["cell", "primary_up"]]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(cold_takeover, FAILOVER_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertIn({"at": 0, "source": "fail"}, raised.exception.context["input"])

        ignores_recovery = copy.deepcopy(FAILOVER_PROGRAM)
        ignores_recovery["handlers"] = [handler for handler in ignores_recovery["handlers"] if handler["on"] != "dispatch.primary-recovered"]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(ignores_recovery, FAILOVER_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertTrue(any(item["source"] == "recover" for item in raised.exception.context["input"]))

    def test_exactly_once_survives_failover_and_separates_operations(self) -> None:
        self.assertEqual(pulse.verify_program(EXACTLY_PROGRAM, EXACTLY_CONTRACT), {
            "program_bytes": 1338, "worst_latency": 0,
            "live_state_cells": 3, "domain_cases": 10417,
        })
        commands = ((0, "A"), (1, "fail"), (2, "A"), (2, "B"), (3, "recover"), (3, "B"))
        expected = pulse.expected_outputs(EXACTLY_CONTRACT, commands)
        self.assertEqual([item["payload"]["source"] for item in expected], ["A", "B"])
        self.assertEqual([item["topic"] for item in expected], ["ring.to-primary", "ring.to-secondary"])

    def test_exactly_once_rejects_forgotten_tombstone_and_conflation(self) -> None:
        route_local = copy.deepcopy(EXACTLY_PROGRAM)
        route_local["handlers"][0]["actions"].append({"op": "set", "cell": "seen_A", "value": ["const", False]})
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(route_local, EXACTLY_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")

        conflated = copy.deepcopy(EXACTLY_PROGRAM)
        conflated["handlers"][3]["actions"][0]["condition"] = ["not", ["cell", "seen_A"]]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(conflated, EXACTLY_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")

    def test_shared_deadline_uses_minimum_absolute_time_and_reopens(self) -> None:
        self.assertEqual(pulse.verify_program(DEADLINE_PROGRAM, DEADLINE_CONTRACT), {
            "program_bytes": 1784, "worst_latency": 3,
            "live_state_cells": 2, "domain_cases": 1471,
        })
        self.assertEqual([item["at"] for item in pulse.expected_outputs(DEADLINE_CONTRACT, ((0, "A"), (0, "B")))], [2])
        self.assertEqual([item["at"] for item in pulse.expected_outputs(DEADLINE_CONTRACT, ((0, "B"), (2, "A"), (3, "A")))], [2, 6])

    def test_shared_deadline_rejects_last_writer_and_stale_timer(self) -> None:
        last_writer = copy.deepcopy(DEADLINE_PROGRAM)
        b_handler = next(handler for handler in last_writer["handlers"] if handler["id"] == "deadline.B")
        b_handler["actions"][1]["condition"] = ["const", False]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(last_writer, DEADLINE_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")

        stale = copy.deepcopy(DEADLINE_PROGRAM)
        for handler in stale["handlers"][:2]:
            for action in handler["actions"]:
                for nested in action.get("actions", []):
                    if nested.get("op") == "schedule":
                        nested["key"] = nested["key"] + handler["id"]
        next(handler for handler in stale["handlers"] if handler["id"] == "deadline.fire")["actions"][0]["condition"] = ["const", True]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(stale, DEADLINE_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")

    def test_warm_failover_routes_each_request_exactly_once(self) -> None:
        duplicates = copy.deepcopy(FAILOVER_PROGRAM)
        duplicates["handlers"][2]["actions"][1]["condition"] = ["cell", "secondary_warm"]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(duplicates, FAILOVER_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertGreater(len(raised.exception.context["observed"]), len(raised.exception.context["expected"]))

    def test_two_of_three_exhausts_sources_duplicates_and_same_tick_order(self) -> None:
        exhaustive = pulse.domain(QUORUM_CONTRACT)
        self.assertEqual(len(exhaustive), 3478)
        duplicate = ((0, "A"), (0, "A"))
        self.assertEqual(pulse.expected_outputs(QUORUM_CONTRACT, duplicate), [])

        same_tick = ((0, "A"), (0, "B"), (0, "C"))
        ordered = pulse.input_events(QUORUM_CONTRACT, same_tick)
        expected = pulse.expected_outputs(QUORUM_CONTRACT, same_tick)
        self.assertEqual(expected, [{
            "at": ordered[1]["at"], "topic": "evac.quorum-release", "payload": ordered[1]["payload"],
        }])

        metrics = pulse.verify_program(QUORUM_PROGRAM, QUORUM_CONTRACT)
        self.assertEqual(metrics, {
            "program_bytes": 1293, "worst_latency": 0,
            "live_state_cells": 4, "domain_cases": 3478,
        })

    def test_two_of_three_rejects_message_counting_and_three_source_wait(self) -> None:
        counting = {
            "format": "afterimage-pulse/0.1",
            "cells": [
                {"name": "count", "type": "int", "initial": 0},
                {"name": "fired", "type": "bool", "initial": False},
            ],
            "handlers": [],
        }
        for source, topic in QUORUM_CONTRACT["input_topics"].items():
            counting["handlers"].append({
                "id": f"counting.on-{source.lower()}", "on": topic,
                "actions": [
                    {"op": "set", "cell": "count", "value": ["add", ["cell", "count"], ["const", 1]]},
                    {"op": "when", "condition": ["and", ["not", ["cell", "fired"]], ["le", ["const", 2], ["cell", "count"]]], "actions": [
                        {"op": "emit", "topic": QUORUM_CONTRACT["output_topic"], "payload": ["event", "payload"]},
                        {"op": "set", "cell": "fired", "value": ["const", True]},
                    ]},
                ],
            })
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(counting, QUORUM_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertEqual(len(raised.exception.context["input"]), 2)
        self.assertEqual(len({item["source"] for item in raised.exception.context["input"]}), 1)

        waits_for_all_three = copy.deepcopy(QUORUM_PROGRAM)
        for handler in waits_for_all_three["handlers"]:
            handler["actions"][1]["condition"][2][0] = "and"
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(waits_for_all_three, QUORUM_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertEqual(len({item["source"] for item in raised.exception.context["input"]}), 2)

    def test_two_of_three_requires_one_shot_and_decisive_payload(self) -> None:
        refires = copy.deepcopy(QUORUM_PROGRAM)
        for handler in refires["handlers"]:
            handler["actions"][1]["actions"].pop()
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(refires, QUORUM_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertGreater(len(raised.exception.context["observed"]), 1)

        wrong_payload = copy.deepcopy(QUORUM_PROGRAM)
        for handler in wrong_payload["handlers"]:
            handler["actions"][1]["actions"][0]["payload"] = ["const", {"sequence": 0, "source": "A"}]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(wrong_payload, QUORUM_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertNotEqual(raised.exception.context["observed"], raised.exception.context["expected"])

    def test_multi_topic_barrier_exhausts_duplicates_order_and_completion(self) -> None:
        exhaustive = pulse.domain(BARRIER_CONTRACT)
        self.assertEqual(len(exhaustive), 3478)
        two_topics = ((0, "wind"), (0, "wind"), (1, "bridge"))
        self.assertEqual(pulse.expected_outputs(BARRIER_CONTRACT, two_topics), [])
        complete = ((0, "hospital"), (1, "wind"), (2, "bridge"))
        ordered = pulse.input_events(BARRIER_CONTRACT, complete)
        self.assertEqual(pulse.expected_outputs(BARRIER_CONTRACT, complete), [{
            "at": 2, "topic": "dawn.barrier-release", "payload": ordered[2]["payload"],
        }])
        self.assertEqual(pulse.verify_program(BARRIER_PROGRAM, BARRIER_CONTRACT), {
            "program_bytes": 1378, "worst_latency": 0,
            "live_state_cells": 4, "domain_cases": 3478,
        })

    def test_multi_topic_barrier_rejects_quorum_and_message_counting(self) -> None:
        quorum_release = copy.deepcopy(BARRIER_PROGRAM)
        for handler in quorum_release["handlers"]:
            handler["actions"][1]["condition"][2][0] = "or"
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(quorum_release, BARRIER_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertEqual(len({item["source"] for item in raised.exception.context["input"]}), 2)

        counts_duplicates = copy.deepcopy(BARRIER_PROGRAM)
        counts_duplicates["cells"] = [
            {"name": "count", "type": "int", "initial": 0},
            {"name": "fired", "type": "bool", "initial": False},
        ]
        counts_duplicates["handlers"] = []
        for source, topic in BARRIER_CONTRACT["input_topics"].items():
            counts_duplicates["handlers"].append({
                "id": f"count.on-{source}", "on": topic,
                "actions": [
                    {"op": "set", "cell": "count", "value": ["add", ["cell", "count"], ["const", 1]]},
                    {"op": "when", "condition": ["and", ["not", ["cell", "fired"]], ["le", ["const", 3], ["cell", "count"]]], "actions": [
                        {"op": "emit", "topic": BARRIER_CONTRACT["output_topic"], "payload": ["event", "payload"]},
                        {"op": "set", "cell": "fired", "value": ["const", True]},
                    ]},
                ],
            })
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(counts_duplicates, BARRIER_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertLess(len({item["source"] for item in raised.exception.context["input"]}), 3)

    def test_multi_topic_barrier_requires_one_shot_and_decisive_payload(self) -> None:
        refires = copy.deepcopy(BARRIER_PROGRAM)
        for handler in refires["handlers"]:
            handler["actions"][1]["actions"].pop()
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(refires, BARRIER_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")
        self.assertGreater(len(raised.exception.context["observed"]), 1)

        wrong_payload = copy.deepcopy(BARRIER_PROGRAM)
        for handler in wrong_payload["handlers"]:
            handler["actions"][1]["actions"][0]["payload"] = ["const", {"sequence": 0, "source": "wind"}]
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(wrong_payload, BARRIER_CONTRACT)
        self.assertEqual(raised.exception.code, "pulse_counterexample")


    def test_sliding_window_exhausts_same_tick_and_expiry_boundary(self) -> None:
        source = json.loads((ROOT / "content/production/cases/PULSE.011.json").read_text(encoding="utf-8"))
        program, contract = source["case"]["author_program"], source["case"]["pulse_contract"]
        self.assertEqual(pulse.verify_program(program, contract), {
            "program_bytes": 478, "worst_latency": 0,
            "live_state_cells": 2, "domain_cases": 792,
        })
        expected = pulse.expected_outputs(contract, (0, 0, 0, 3))
        self.assertEqual([item["payload"]["sequence"] for item in expected], [0, 2, 3])

    def test_sliding_window_rejected_request_is_not_remembered(self) -> None:
        source = json.loads((ROOT / "content/production/cases/PULSE.011.json").read_text(encoding="utf-8"))
        program, contract = source["case"]["author_program"], source["case"]["pulse_contract"]
        charges_rejected = copy.deepcopy(program)
        guarded = charges_rejected["handlers"][0]["actions"][0]
        shifts = guarded["actions"][1:]
        charges_rejected["handlers"][0]["actions"].extend(copy.deepcopy(shifts))
        with self.assertRaises(pulse.PulseError) as raised:
            pulse.verify_program(charges_rejected, contract)
        self.assertEqual(raised.exception.code, "pulse_counterexample")


if __name__ == "__main__":
    unittest.main(verbosity=2)
