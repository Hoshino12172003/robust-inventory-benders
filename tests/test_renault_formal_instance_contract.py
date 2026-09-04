from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from src.instance import InventoryInstance, load_instance
from src.renault_formal_dry_run import DEFAULT_DATA_ROOT, run_dry_run


def test_inventory_instance_loads_all_formal_instances_without_a_k_field() -> None:
    instance_paths = sorted((DEFAULT_DATA_ROOT / "instances").glob("*_B*.json"))
    assert len(instance_paths) == 8
    assert "K" not in {field.name for field in fields(InventoryInstance)}
    for path in instance_paths:
        instance = load_instance(path)
        assert (instance.num_warehouses, instance.num_products, instance.num_regions) == (15, 8, 12)


def test_renault_step4a_contract(tmp_path: Path) -> None:
    report = run_dry_run(DEFAULT_DATA_ROOT, tmp_path)
    assert report["overall"] == "PASS"
    assert report["package_hashes"] == "PASS"
    assert report["formal_gamma2_optimization_executed"] is False
    assert report["core_mathematical_model_changed"] is False
    assert len(report["instances"]) == 8
    assert all(item["overall"] == "PASS" for item in report["instances"])
