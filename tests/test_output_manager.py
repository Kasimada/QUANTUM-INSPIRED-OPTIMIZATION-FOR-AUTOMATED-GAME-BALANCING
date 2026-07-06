import os
import shutil
import pytest
from pathlib import Path
from src.utils.output_manager import OutputManager, OutputLayout

def test_flat_collision(tmp_path):
    base_dir = tmp_path / "result"
    
    # 1st run
    out1 = OutputManager(str(base_dir), OutputLayout.FLAT, {"mode": "continuous"})
    col1 = out1.detect_collision()
    assert not col1.exists
    
    with out1 as info1:
        assert info1.absolute_path == str(base_dir)
        
    # 2nd run
    out2 = OutputManager(str(base_dir), OutputLayout.FLAT, {"mode": "continuous"})
    col2 = out2.detect_collision()
    assert col2.exists
    assert col2.has_manifest
    assert col2.status == "completed"

def test_date_config_auto_increment(tmp_path):
    base_dir = tmp_path / "result"
    
    # Run 1
    with OutputManager(str(base_dir), OutputLayout.DATE_CONFIG, {"mode": "continuous"}) as info1:
        assert "001_continuous" in info1.absolute_path
        
    # Run 2
    with OutputManager(str(base_dir), OutputLayout.DATE_CONFIG, {"mode": "continuous"}) as info2:
        assert "002_continuous" in info2.absolute_path

def test_100_runs_no_collision(tmp_path):
    """Health check: Create 100 runs in a row to ensure 001_ to 100_ works correctly."""
    base_dir = tmp_path / "result_100"
    for i in range(1, 101):
        with OutputManager(str(base_dir), OutputLayout.CONFIG, {"mode": "discrete"}) as info:
            folder_name = Path(info.absolute_path).name
            expected_prefix = f"{i:03d}_discrete"
            assert folder_name == expected_prefix

def test_failed_status_on_exception(tmp_path):
    base_dir = tmp_path / "result_fail"
    
    try:
        with OutputManager(str(base_dir), OutputLayout.FLAT, {"mode": "test"}) as info:
            raise ValueError("Something broke!")
    except ValueError:
        pass
        
    # Check if manifest recorded failed status
    import json
    man_path = Path(info.absolute_path) / "manifest.json"
    with open(man_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["status"] == "failed"
        assert not data["is_completed"]

def test_lock_file(tmp_path):
    base_dir = tmp_path / "result_lock"
    
    out = OutputManager(str(base_dir), OutputLayout.FLAT, {"mode": "test"})
    out.prepare() # Normally called by __enter__
    
    # Simulate a crash where __exit__ isn't called
    with open(Path(out.out_info.absolute_path) / ".running.lock", "w") as f:
        f.write('{"pid": 999999}')
        
    out2 = OutputManager(str(base_dir), OutputLayout.FLAT, {"mode": "test"})
    col = out2.detect_collision()
    assert col.exists
    assert col.has_lock
    assert col.pid == 999999
