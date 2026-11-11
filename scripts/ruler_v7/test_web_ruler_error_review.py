"""Persistence and read-only-asset checks with synthetic fixtures, no models."""

import csv
import hashlib
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from web_ruler_error_review import ReviewInputs, ReviewServer, ReviewStore


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ruler-web-test-", dir="/tmp/opencode")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets, self.reports = self.root / "assets", self.root / "reports"
        self.assets.mkdir()
        Image.new("RGB", (500, 400), "gray").save(self.assets / "source.png")
        Image.new("RGB", (40, 50), "white").save(self.assets / "crop.png")
        Image.new("L", (40, 50), 255).save(self.assets / "mask.png")
        self.review = self.reports / "manual_review"
        self.review.mkdir(parents=True)
        self.write_csv("manual_review/index.csv", [
            dict(source_id="source", detection_id=str(i), original_path="crop.png",
                 v5_mask_path="mask.png", v7_mask_path="mask.png" if i == 1 else "",
                 v5_has_mask="True", v7_has_mask="True" if i == 1 else "False",
                 difference_review_candidate="True", bbox_x1="100.25", bbox_y1="120.75",
                 bbox_x2="140", bbox_y2="170") for i in (1, 2)
        ])
        self.write_csv("manual_review/ruler_position_review.csv", [
            dict(source_id="source", detection_id=str(i), position_class="R", reviewed_at="original-time")
            for i in (1, 2)
        ])
        self.write_csv("manual_review/manual_review_decisions.csv", [
            dict(source_id="source", detection_id=str(i), decision="BOTH_BAD", reviewed_at="old-rating-time")
            for i in (1, 2)
        ])
        self.write_csv("manual_review/both_bad_reason_review.csv", [
            dict(source_id="source", detection_id="1", reason="BODY_PART_MISSING", reviewed_at="old-reason-time")
        ])
        self.write_csv("unseen/candidate_selection_leakage_audit.csv", [
            dict(source_id="source", image_path="source.png")
        ])
        self.inputs = ReviewInputs(self.reports, self.assets, expected_total=2, expected_cases=2)
        self.before = self.hash_inputs()
        self.output = self.root / "new_decisions"
        self.store = ReviewStore(self.inputs, self.output)

    def write_csv(self, name, rows):
        path = self.reports / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def hash_inputs(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for directory in (self.assets, self.reports)
                for p in directory.rglob("*") if p.is_file()}

    def test_prior_reason_is_not_inferred_as_new_per_model_rating(self):
        self.assertEqual(self.inputs.cases[0]["detection_id"], "2")
        self.assertEqual(self.inputs.cases[1]["prior_reason"]["reviewed_at"], "old-reason-time")
        self.assertEqual(self.store.state()["next_pending"], 0)
        self.assertFalse(self.store.state()["decisions"][1]["v5"])

    def test_autosave_resume_and_revision_history(self):
        self.store.record(0, "v5", "TOO_SMALL")
        restarted = ReviewStore(self.inputs, self.output)
        first = restarted.state()["decisions"][0]["v5"]
        restarted.record(0, "v5", "TOO_SMALL")
        self.assertEqual(restarted.state()["decisions"][0]["v5"], first)
        restarted.record(0, "v5", "RULER_INCLUDED")
        with sqlite3.connect(restarted.database) as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM events").fetchone()[0], 2)
            self.assertEqual(con.execute("SELECT reviewed_at FROM events ORDER BY id LIMIT 1").fetchone()[0], first["reviewed_at"])
        restarted.record(0, "v7", "BODY_PART_MISSING")
        self.assertEqual(restarted.state()["completed"], 0)
        restarted.record(0, "crop", "COMPLETE")
        self.assertEqual(restarted.state()["next_pending"], 1)
        self.assertEqual(self.hash_inputs(), self.before)

    def test_summary_grouping_and_no_premature_recommendation(self):
        self.assertFalse(self.store.summary()["finished"])
        for index in range(2):
            self.store.record(index, "crop", "TRUNCATED" if index == 0 else "COMPLETE")
            self.store.record(index, "v5", "TOO_SMALL")
            self.store.record(index, "v7", "RULER_INCLUDED")
        summary = self.store.summary()
        self.assertTrue(summary["finished"])
        self.assertIn("Kombinacija", summary["next_step"])
        for model in ("v5", "v7"):
            rows = [r for r in summary["counts_by_source_model_crop_and_error"]
                    if r["source_id"] == "ALL" and r["model"] == model]
            self.assertEqual(sum(r["count"] for r in rows), 2)
        self.assertTrue((self.output / "summary_by_source_model_error.csv").is_file())

    def test_invalid_values_do_not_save(self):
        for args in ((-1, "v5", "TOO_SMALL"), (0, "v8", "TOO_SMALL"),
                     (0, "crop", "TOO_SMALL"), (True, "v5", "TOO_SMALL")):
            with self.assertRaises(ValueError):
                self.store.record(*args)
        self.assertEqual(self.store.latest(), {})

    def test_truncated_inputs_do_not_alone_justify_training(self):
        for index in range(2):
            self.store.record(index, "crop", "TRUNCATED")
            self.store.record(index, "v5", "BODY_PART_MISSING")
            self.store.record(index, "v7", "TOO_SMALL")
        self.assertIn("crop/padding", self.store.summary()["next_step"])
        self.assertNotIn("fine-tuniranje", self.store.summary()["next_step"])

    def test_unassessable_masks_do_not_invent_annotation_recommendation(self):
        for index in range(2):
            self.store.record(index, "crop", "COMPLETE")
            self.store.record(index, "v5", "UNASSESSABLE")
            self.store.record(index, "v7", "UNASSESSABLE")
        self.assertIn("razjasniti", self.store.summary()["next_step"])
        self.assertEqual(self.store.summary()["evidence"]["complete_body_model_errors"], 0)

    def test_snapshot_and_reports_cannot_be_write_destinations(self):
        for directory in (self.assets / "new", self.reports / "new"):
            with self.assertRaises(ValueError):
                ReviewStore(self.inputs, directory)
            self.assertFalse(directory.exists())
        with self.assertRaises(ValueError):
            self.inputs.asset("../outside.png")

    def test_changed_input_cannot_reuse_old_decisions(self):
        self.store.record(0, "v5", "MULTIPLE_PROBLEMS")
        self.inputs.fingerprint = "different-input-set"
        with self.assertRaises(ValueError):
            ReviewStore(self.inputs, self.output)

    def test_context_uses_saved_crop_origin_not_rounded_bbox(self):
        box = self.inputs.context_box(self.inputs.cases[0])
        self.assertEqual(box["box"], [0, 0, 500, 400])
        self.assertEqual(box["crop_box"], [100, 120, 140, 170])

    def test_http_assets_and_durable_post(self):
        server = ReviewServer(("127.0.0.1", 0), self.inputs, self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/state") as response:
                self.assertEqual(len(json.load(response)["cases"]), 2)
            with urlopen(base + "/asset?index=0&kind=original") as response:
                self.assertEqual(response.read(), (self.assets / "crop.png").read_bytes())
            with self.assertRaises(HTTPError) as missing:
                urlopen(base + "/asset?index=0&kind=v7")
            self.assertEqual(missing.exception.code, 404)
            with urlopen(base + "/asset?index=0&kind=context") as response:
                with Image.open(io.BytesIO(response.read())) as image:
                    self.assertEqual(image.size, (500, 400))
            req = Request(base + "/api/decision", data=json.dumps({
                "case": 0, "model": "v5", "category": "TOO_LARGE"
            }).encode(), headers={"Content-Type": "application/json"})
            with urlopen(req) as response:
                self.assertEqual(json.load(response)["state"]["decisions"][0]["v5"]["category"], "TOO_LARGE")
            self.assertEqual(ReviewStore(self.inputs, self.output).state()["decisions"][0]["v5"]["category"], "TOO_LARGE")
            self.assertEqual(self.hash_inputs(), self.before)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
