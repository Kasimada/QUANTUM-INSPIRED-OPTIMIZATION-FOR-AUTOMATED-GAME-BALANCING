import json
from pathlib import Path

class ReportGenerator:
    """
    Generates presentation artifacts (README, etc.) from the raw data
    produced by the experiment (manifest, CSVs).
    """

    @staticmethod
    def generate(output_dir: Path) -> None:
        manifest_file = output_dir / "manifest.json"
        if not manifest_file.exists():
            return

        with open(manifest_file, "r", encoding="utf-8") as f:
            try:
                manifest = json.load(f)
            except json.JSONDecodeError:
                return

        lines = [
            f"# Experiment Run: {manifest.get('mode', 'unknown').upper()}",
            "",
            f"**Created**: {manifest.get('created_at', 'N/A')}",
            f"**Status**: {manifest.get('status', 'unknown')}",
            "",
            "## Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Mode | {manifest.get('mode', 'N/A')} |",
            f"| Trials | {manifest.get('trials', 'N/A')} |",
            f"| FEs | {manifest.get('fes', 'N/A')} |",
            f"| Dataset | {manifest.get('dataset', 'N/A')} |",
            f"| Scenario | {manifest.get('scenario', 'symmetric')} |",
            f"| Seed | {manifest.get('seed', 'N/A')} |",
            f"| Workers | {manifest.get('workers', 'N/A')} |",
            f"| Device | {manifest.get('device', 'cpu')} |",
            "",
            "## Algorithms",
            "",
        ]
        
        for alg in manifest.get("algorithms", []):
            lines.append(f"- {alg}")
        
        lines.append("")
        lines.append("## Environment")
        lines.append("")
        
        sys_info = manifest.get("system_info", {})
        lines.append(f"- Python: {sys_info.get('python', 'N/A')}")
        lines.append(f"- Platform: {sys_info.get('platform', 'N/A')} ({sys_info.get('os', 'N/A')})")
        lines.append(f"- CPU: {sys_info.get('cpu', 'N/A')} ({sys_info.get('cores', 'N/A')} cores)")
        lines.append(f"- Hostname: {sys_info.get('hostname', 'N/A')}")
        lines.append(f"- Machine: {sys_info.get('machine_model', 'N/A')}")
        
        if "git_commit" in manifest:
            lines.append(f"- Git Commit: `{manifest['git_commit']}`")
            
        if "completed_at" in manifest:
            lines.append("")
            lines.append(f"**Completed**: {manifest['completed_at']}")
            if "duration" in manifest:
                lines.append(f"**Duration**: {manifest['duration']:.2f} seconds")
                
        lines.append("")

        with open(output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
