"""Synthetic test: patient-level split never leaks. No real data needed."""
from __future__ import annotations

from src.dataset import SliceRecord, patient_level_split, verify_no_patient_leakage


def _make_records(n_patients: int = 30, slices_per_patient: int = 20) -> list[SliceRecord]:
    recs = []
    for p in range(n_patients):
        pid = f"TCGA_DUMMY_{p:04d}"
        for s in range(slices_per_patient):
            recs.append(SliceRecord(
                patient_id=pid,
                image_path=f"/fake/{pid}/{s}.tif",
                mask_path=f"/fake/{pid}/{s}_mask.tif",
                slice_idx=s,
                has_tumor=(s % 3 == 0),
            ))
    return recs


def test_split_disjoint_patients():
    records = _make_records()
    splits = patient_level_split(records, val_frac=0.2, test_frac=0.2, seed=0)
    train_ids = {r.patient_id for r in splits["train"]}
    val_ids = {r.patient_id for r in splits["val"]}
    test_ids = {r.patient_id for r in splits["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    verify_no_patient_leakage(splits)


def test_split_covers_all_records():
    records = _make_records()
    splits = patient_level_split(records, seed=1)
    total = sum(len(v) for v in splits.values())
    assert total == len(records)


def test_seed_is_deterministic():
    records = _make_records()
    a = patient_level_split(records, seed=7)
    b = patient_level_split(records, seed=7)
    assert {r.patient_id for r in a["test"]} == {r.patient_id for r in b["test"]}


if __name__ == "__main__":
    test_split_disjoint_patients()
    test_split_covers_all_records()
    test_seed_is_deterministic()
    print("All split tests passed.")
