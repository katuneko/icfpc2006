#!/usr/bin/env python3
"""Integration and adversarial tests for production wave 19."""

from __future__ import annotations
import copy, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "reference/python"))
import afterimage_kit as kit  # noqa: E402
import cre  # noqa: E402
import pulse  # noqa: E402
import verify_witness as verifier  # noqa: E402


class ProductionWave19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix="afterimage-wave19-")
        cls.root=Path(cls.temporary.name); cls.bundle=cls.root/"production.afterimage"; cls.author=cls.root/"author"
        subprocess.run([sys.executable,str(ROOT/"tools/build_slice.py"),str(cls.bundle),"--manifest",str(ROOT/"manifests/production_release.json"),"--author-dir",str(cls.author),"--title","Afterimage production release 2.0","--revision","production-dev-2.0.0"],check=True,capture_output=True,text=True)
        cls.world_path=cls.root/"world"; kit.extract_bundle(kit.load_bundle(cls.bundle),cls.world_path); cls.world=kit.verify_world(cls.world_path)
    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()
    def witness(self,c): return json.loads((self.author/f"{c}.witness.json").read_text())
    def case(self,c):
        d=next(x for x in self.world.json_values["cases/index.json"]["cases"] if x["id"]==c)
        return verifier.validate_case_descriptor(d,self.world)
    def verify(self,c,name,w,facts):
        w.pop("meta",None); p=self.root/f"{c}-{name}.json"; p.write_bytes(cre.canonical_bytes(w)); return verifier.verify_witness(self.world_path,p,facts)

    def test_all_five_author_baselines_are_valid(self):
        for c,s in {"MERGE.014":129,"PULSE.013":149,"CASCADE.020":99,"CASCADE.021":109,"CASCADE.022":109}.items():
            r=json.loads((self.author/f"{c}.receipt.json").read_text()); self.assertTrue(r["valid"],c); self.assertEqual(r["score"]["total"],s,c)

    def test_non_unique_archive_rejects_noncanonical_valid_schedule(self):
        w=self.witness("MERGE.014"); middle=w["answer"]["accepted"][1]; middle["at"]=12
        r=self.verify("MERGE.014","later",w,{"case:MERGE.013","case:CASCADE.019"})
        self.assertFalse(r["valid"]); self.assertEqual(r["diagnostics"][0]["code"],"merge_non_unique")

    def test_circuit_breaker_exhausts_half_open_state(self):
        s=json.loads((ROOT/"content/production/cases/PULSE.013.json").read_text())["case"]
        self.assertEqual(pulse.verify_program(s["author_program"],s["pulse_contract"]),{"program_bytes":1664,"worst_latency":2,"live_state_cells":3,"domain_cases":3478})
        broken=copy.deepcopy(s["author_program"]); broken["handlers"][0]["actions"][1]["actions"]=broken["handlers"][0]["actions"][1]["actions"][:1]
        with self.assertRaises(pulse.PulseError) as raised: pulse.verify_program(broken,s["pulse_contract"])
        self.assertEqual(raised.exception.code,"pulse_counterexample")

    def test_competing_amendments_bind_public_canonical_choice(self):
        case=self.case("CASCADE.020"); root=verifier.replay_root_case(self.world,case)
        fixed=self.witness("CASCADE.020"); op=fixed["intervention"]["operations"][0]
        replay=verifier.replay_case(self.world,case,[op],cre.root_branch_id(self.world.bundle))
        self.assertFalse(root.records[0]["safe"]); self.assertTrue(replay.records[0]["safe"])
        self.assertEqual(replay.records[0]["chosen"],"audit-first")
        self.assertEqual(replay.records[0]["alternatives_preserved"],["audit-first","evidence-first"])

    def test_audit_window_uses_inclusive_closing_tick(self):
        case=self.case("CASCADE.021"); root=verifier.replay_root_case(self.world,case)
        w=self.witness("CASCADE.021"); replay=verifier.replay_case(self.world,case,w["intervention"]["operations"],cre.root_branch_id(self.world.bundle))
        self.assertFalse(root.records[0]["safe"]); self.assertTrue(replay.records[0]["safe"]); self.assertEqual(replay.records[0]["filed_at"],22)

    def test_least_change_beats_valid_two_operation_repair(self):
        case=self.case("CASCADE.022"); logical=kit.resolve_logical_world(self.world,case["world"])
        disclosure=next(e for e in logical.base_events if e["topic"]=="hearing.disclosure")
        audit=next(e for e in logical.base_events if e["topic"]=="hearing.audit-time"); parent=cre.root_branch_id(self.world.bundle)
        operations=[{"kind":"replace","event":disclosure["id"],"pointer":"/payload/disclosure","value":"public"},{"kind":"retime","event":audit["id"],"at":22}]
        replay=verifier.replay_case(self.world,case,operations,parent)
        self.assertTrue(replay.records[0]["safe"])
        w=self.witness("CASCADE.022"); w["intervention"]["operations"]=operations; w["answer"]={"contracts":replay.records,"branch":replay.branch,"projection":replay.projection}; w["claimed"]={"branch":replay.branch,"projection":replay.projection,"trace":replay.trace}
        r=self.verify("CASCADE.022","costly",w,{"case:CASCADE.021"}); self.assertTrue(r["valid"])
        author=json.loads((self.author/"CASCADE.022.receipt.json").read_text())
        self.assertGreater(r["metrics"]["effective_cost"],author["metrics"]["effective_cost"])


if __name__=="__main__": unittest.main(verbosity=2)
