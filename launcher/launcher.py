import sys
import json
import time
import platform
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algorithms.factory import get_supported_algorithms
from src.simulation.type_chart import get_supported_scenarios
from src.version import APP_NAME, __version__
from src.utils.output_manager import OutputManager, OutputLayout

LANG = 'en'
CONFIG_DIR = Path("configs/launcher_profiles")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

TEXT = {
    'en': {
        'title': f'=== {APP_NAME} v{__version__} ===',
        'lang_prompt': 'Select Language (1=En, 2=Vi): ',
        'menu': '\n1. Dataset\n2. Mode\n3. Scenario\n4. Algorithms\n5. Trials\n6. FEs\n7. Workers\n8. Device\n9. Seed\n10. Output Folder\n11. Layout\n12. Load Profile\n13. Save Profile\n14. Export Config\n15. Import Config\n16. Delete Profile\n\n0. Review Configuration\nSelect option (B=Back, Q=Quit, S=Start, H=Help): ',
        'not_set': 'Not Selected',
        'conf_title': '\n=========================================\nConfiguration Summary\n=========================================',
        'status_incomplete': 'Status: Configuration Incomplete',
        'status_ready': 'Status: Ready to Run',
        'dataset_title': 'Available Datasets:',
        'dataset_prompt': 'Select dataset (0 to cancel): ',
        'mode_prompt': '\n1. continuous\n2. discrete\n3. both\nSelect mode: ',
        'scenario_prompt': 'Enter scenario name (or leave blank for symmetric): ',
        'alg_prompt': 'Enter number to toggle, or type "done" (or "all" for all): ',
        'trials_prompt': 'Enter number of trials (e.g. 30): ',
        'fes_prompt': 'Enter max FEs (e.g. 50000): ',
        'workers_prompt': 'Enter number of CPU workers: ',
        'device_prompt': '\n1. cpu\n2. auto (will use gpu if available)\nSelect device: ',
        'seed_prompt': 'Enter random seed (blank for default): ',
        'out_prompt': 'Enter output folder path (blank for "results/"): ',
        'layout_prompt': '\n1. date_config\n2. date\n3. config\n4. flat\nSelect output layout: ',
        'save_prompt': 'Enter profile name to save: ',
        'save_desc': 'Enter profile description: ',
        'load_title': 'Available Profiles:',
        'load_prompt': 'Select profile number (0 to cancel): ',
        'delete_title': 'Available Profiles to Delete:',
        'delete_prompt': 'Select profile number to delete (0 to cancel): ',
        'review_start': '\nS = Start\nE = Edit Configuration\nQ = Quit\nYour choice: ',
        'invalid': 'Invalid choice.',
        'help': '\n--- HELP ---\nB = Go back/Cancel current action.\nQ = Quit Launcher.\nS = Start experiment.\nC = Cancel running experiment.',
        'stale_lock': 'Stale lock detected. Remove automatically? (Y/N): ',
        'interrupted': 'Interrupted experiment detected!\n1. Restart here (overwrite)\n2. Archive old run and create new\n3. Open folder\n4. Cancel\nYour choice: ',
        'collision': 'Folder already exists and contains data!\n1. Overwrite\n2. Cancel\nYour choice: ',
        'export_prompt': 'Enter export filename (e.g. benchmark.json): '
    },
    'vi': {
        'title': f'=== Khởi Chạy {APP_NAME} v{__version__} ===',
        'lang_prompt': 'Chọn Ngôn ngữ (1=En, 2=Vi): ',
        'menu': '\n1. Dataset\n2. Mode\n3. Scenario\n4. Algorithms\n5. Trials\n6. FEs\n7. Workers\n8. Device\n9. Seed\n10. Output Folder\n11. Layout\n12. Load Profile\n13. Save Profile\n14. Export Config\n15. Import Config\n16. Delete Profile\n\n0. Xem lại Cấu Hình\nChọn tuỳ chọn (B=Quay lại, Q=Thoát, S=Bắt đầu chạy, H=Trợ giúp): ',
        'not_set': 'Chưa chọn',
        'conf_title': '\n=========================================\nCấu Hình Hiện Tại\n=========================================',
        'status_incomplete': 'Trạng thái: Chưa hoàn tất cấu hình',
        'status_ready': 'Trạng thái: Đã sẵn sàng chạy',
        'dataset_title': 'Các Dataset hiện có:',
        'dataset_prompt': 'Chọn Dataset (0 = huỷ): ',
        'mode_prompt': '\n1. continuous (Liên tục)\n2. discrete (Rời rạc)\n3. both (Cả hai)\nChọn Mode: ',
        'scenario_prompt': 'Nhập tên kịch bản (để trống = symmetric): ',
        'alg_prompt': 'Nhập số để bật/tắt, hoặc "done" để xong (hoặc "all" chọn tất cả): ',
        'trials_prompt': 'Nhập số lần chạy (Trials, vd: 30): ',
        'fes_prompt': 'Nhập số đánh giá tối đa (FEs, vd: 50000): ',
        'workers_prompt': 'Nhập số luồng CPU (Workers): ',
        'device_prompt': '\n1. cpu\n2. auto (ưu tiên gpu nếu có)\nChọn thiết bị: ',
        'seed_prompt': 'Nhập seed (để trống = mặc định): ',
        'out_prompt': 'Nhập thư mục lưu kết quả (để trống = "results/"): ',
        'layout_prompt': '\n1. date_config\n2. date\n3. config\n4. flat\nChọn kiểu sắp xếp thư mục: ',
        'save_prompt': 'Nhập tên Profile để lưu: ',
        'save_desc': 'Mô tả Profile: ',
        'load_title': 'Các Profile đã lưu:',
        'load_prompt': 'Chọn Profile (0 = huỷ): ',
        'delete_title': 'Các Profile có thể xoá:',
        'delete_prompt': 'Chọn Profile để xoá (0 = huỷ): ',
        'review_start': '\nS = Bắt đầu (Start)\nE = Sửa cấu hình (Edit)\nQ = Thoát (Quit)\nLựa chọn: ',
        'invalid': 'Không hợp lệ.',
        'help': '\n--- TRỢ GIÚP ---\nB = Quay lại bước trước.\nQ = Thoát chương trình.\nS = Bắt đầu chạy.\nC = Hủy ngang khi đang chạy.',
        'stale_lock': 'Phát hiện tiến trình cũ bị treo (stale lock). Xóa tự động? (Y/N): ',
        'interrupted': 'Phát hiện thực nghiệm đang chạy dở!\n1. Chạy lại từ đầu (ghi đè)\n2. Lưu trữ (Archive) bản cũ và tạo bản mới\n3. Mở thư mục\n4. Hủy\nLựa chọn: ',
        'collision': 'Thư mục đã có dữ liệu!\n1. Ghi đè\n2. Hủy\nLựa chọn: ',
        'export_prompt': 'Tên file xuất ra (vd benchmark.json): '
    }
}

config = {
    'dataset': None,
    'mode': None,
    'scenario': 'symmetric',
    'algorithms': [], 
    'algorithms_continuous': [],
    'algorithms_discrete': [],
    'trials': 30,
    'fes': 50000,
    'workers': min(6, max(1, multiprocessing.cpu_count() - 2)),
    'device': 'auto',
    'seed': 2026,
    'output': '',
    'layout': 'date_config',
    'profile_name': 'default'
}

def t(key):
    return TEXT[LANG].get(key, key)

def safe_input(prompt_text):
    while True:
        ans = input(prompt_text).strip()
        al = ans.lower()
        if al == 'q':
            print("Exiting launcher...")
            sys.exit(0)
        return ans

def select_dataset():
    data_dir = Path("data")
    all_csvs = list(data_dir.rglob("*.csv")) if data_dir.exists() else []
    csvs = [p for p in all_csvs if p.name not in ('seed_puuids.csv', 'vn2_champion_rates.csv')]
    if not csvs:
        return
    print(f"\n{t('dataset_title')}")
    for i, p in enumerate(csvs, 1):
        print(f"{i}. {p.name}")
    ans = safe_input(t('dataset_prompt'))
    if ans.lower() == 'b': return
    if ans.isdigit():
        idx = int(ans)
        if 1 <= idx <= len(csvs):
            config['dataset'] = str(csvs[idx-1])
        else:
            print(t('invalid'))

def select_mode():
    ans = safe_input(t('mode_prompt'))
    if ans.lower() == 'b': return
    if ans == '1': config['mode'] = 'continuous'
    elif ans == '2': config['mode'] = 'discrete'
    elif ans == '3': config['mode'] = 'both'
    else: print(t('invalid'))

def select_scenario():
    print(t('scenario_prompt'))
    scens = get_supported_scenarios()
    for i, s in enumerate(scens, 1):
        print(f"{i}. {s}")
    ans = safe_input("Your choice: ")
    if ans.lower() == 'b': return
    if ans.isdigit():
        idx = int(ans)
        if 1 <= idx <= len(scens):
            config['scenario'] = scens[idx-1]
        else:
            print(t('invalid'))

def checklist(title, options, selected):
    while True:
        print(f"\n--- {title} ---")
        for i, opt in enumerate(options, 1):
            mark = 'x' if opt in selected else ' '
            print(f"{i}. [{mark}] {opt}")
        ans = safe_input(t('alg_prompt'))
        if ans.lower() == 'b' or ans.lower() == 'done': break
        if ans.lower() == 'all':
            selected.clear()
            selected.extend(options)
        elif ans.isdigit():
            idx = int(ans)
            if 1 <= idx <= len(options):
                opt = options[idx-1]
                if opt in selected: selected.remove(opt)
                else: selected.append(opt)
            else:
                print(t('invalid'))

def select_algorithms():
    if not config['mode']: return
    if config['mode'] == 'both':
        checklist("Continuous Algorithms", get_supported_algorithms('continuous'), config['algorithms_continuous'])
        checklist("Discrete Algorithms", get_supported_algorithms('discrete'), config['algorithms_discrete'])
    else:
        checklist(f"{config['mode'].capitalize()} Algorithms", get_supported_algorithms(config['mode']), config['algorithms'])

def select_integer(key, prompt_key, default_val):
    ans = safe_input(t(prompt_key))
    if ans.lower() == 'b': return
    if ans == '': config[key] = default_val
    elif ans.isdigit():
        val = int(ans)
        if key == 'workers':
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            if val > cpu_count:
                print(f"⚠️  WARNING: Requested workers ({val}) > Available cores ({cpu_count}). Capping at {cpu_count}.")
                val = cpu_count
            
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
                max_safe_workers = max(1, int(ram_gb // 2)) # Assume ~2GB per worker
                if val > max_safe_workers:
                    print(f"\n⚠️  CRITICAL WARNING: You have {ram_gb:.1f}GB RAM. {val} workers will likely cause an Out-Of-Memory CRASH!")
                    safe_ans = input(f"Auto-reduce to safe limit ({max_safe_workers})? [Y/n]: ")
                    if safe_ans.lower() != 'n':
                        val = max_safe_workers
                        print(f"Workers safely reduced to {val}.")
            except ImportError:
                pass
                
        config[key] = val
    else: print(t('invalid'))

def select_device():
    import torch
    has_cuda = torch.cuda.is_available()
    opts = ["auto", "cpu"]
    if has_cuda:
        opts.append("cuda")
        for i in range(torch.cuda.device_count()):
            opts.append(f"cuda:{i}")
            
    print(t('device_prompt'))
    if has_cuda:
        for i, o in enumerate(opts, 1):
            print(f"{i}. {o}")
        ans = safe_input("Your choice: ")
        if ans.lower() == 'b': return
        if ans.isdigit():
            idx = int(ans)
            if 1 <= idx <= len(opts):
                config['device'] = opts[idx-1]
    else:
        ans = safe_input("Your choice: ")
        if ans.lower() == 'b': return
        if ans == '1' or ans == '2':
            config['device'] = 'cpu'

def select_output():
    ans = safe_input(t('output_prompt'))
    if ans.lower() == 'b': return
    config['output'] = ans

def select_layout():
    ans = safe_input(t('layout_prompt'))
    if ans.lower() == 'b': return
    if ans == '1': config['layout'] = 'date_config'
    elif ans == '2': config['layout'] = 'date'
    elif ans == '3': config['layout'] = 'config'
    elif ans == '4': config['layout'] = 'flat'

def save_config():
    name = safe_input(t('save_name'))
    if name.lower() == 'b': return
    if not name: return
    desc = safe_input(t('save_desc'))
    if not name.endswith('.json'): name += '.json'
    
    meta = {
        "profile_name": name.replace('.json', ''),
        "description": desc,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "launcher_version": __version__,
        "schema_version": 1
    }
    data = dict(config)
    data["metadata"] = meta
    with open(CONFIG_DIR / name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    config['profile_name'] = meta["profile_name"]

def load_config():
    profiles = list(CONFIG_DIR.glob("*.json"))
    if not profiles: return
    print(f"\n{t('load_title')}")
    for i, p in enumerate(profiles, 1):
        print(f"{i}. {p.name}")
    ans = safe_input(t('load_prompt'))
    if ans.lower() == 'b': return
    if ans.isdigit():
        idx = int(ans)
        if idx == 0: return
        if 1 <= idx <= len(profiles):
            with open(profiles[idx-1], "r", encoding="utf-8") as f:
                data = json.load(f)
                config.update({k: v for k, v in data.items() if k != "metadata"})
                config["profile_name"] = data.get("metadata", {}).get("profile_name", profiles[idx-1].stem)
        else:
            print(t('invalid'))

def delete_config():
    profiles = list(CONFIG_DIR.glob("*.json"))
    if not profiles: return
    print(f"\n{t('delete_title')}")
    for i, p in enumerate(profiles, 1):
        print(f"{i}. {p.name}")
    ans = safe_input(t('delete_prompt'))
    if ans.lower() == 'b': return
    if ans.isdigit():
        idx = int(ans)
        if idx == 0: return
        if 1 <= idx <= len(profiles):
            target = profiles[idx-1]
            target.unlink(missing_ok=True)
            print(f"Deleted profile: {target.name}")
            if config.get("profile_name") == target.stem:
                config["profile_name"] = "default"
        else:
            print(t('invalid'))

def export_config():
    ans = safe_input(t('export_prompt'))
    if ans.lower() == 'b': return
    if not ans: return
    if not ans.endswith('.json'): ans += '.json'
    with open(ans, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"Exported to {ans}")

def import_config():
    ans = safe_input("Enter path to JSON config file to import: ")
    if ans.lower() == 'b': return
    if not ans: return
    try:
        with open(ans, "r", encoding="utf-8") as f:
            data = json.load(f)
            config.update({k: v for k, v in data.items() if k != "metadata"})
            config["profile_name"] = data.get("metadata", {}).get("profile_name", Path(ans).stem)
        print(f"Successfully imported config from {ans}")
    except Exception as e:
        print(f"Failed to import config: {e}")

def is_ready():
    req = ['dataset', 'mode', 'scenario', 'trials', 'fes', 'workers', 'device']
    for r in req:
        if config[r] is None: return False
    if config['mode'] == 'both':
        if not config['algorithms_continuous'] and not config['algorithms_discrete']: return False
    else:
        if not config['algorithms']: return False
    return True

def render_state():
    print(t('conf_title'))
    print(f"Language      : {'English' if LANG=='en' else 'Tiếng Việt'}")
    print(f"Profile       : {config.get('profile_name', 'default')}")
    print(f"Dataset       : {config['dataset'] or t('not_set')}")
    print(f"Mode          : {config['mode'] or t('not_set')}")
    print(f"Scenario      : {config['scenario'] or t('not_set')}")
    if config['mode'] == 'both':
        print(f"Algorithms (C): {','.join(config['algorithms_continuous']) or t('not_set')}")
        print(f"Algorithms (D): {','.join(config['algorithms_discrete']) or t('not_set')}")
    else:
        print(f"Algorithms    : {','.join(config['algorithms']) or t('not_set')}")
        
    print(f"Trials        : {config['trials']}")
    print(f"FEs           : {config['fes']}")
    print(f"Workers       : {config['workers']}")
    print(f"Device        : {config['device']}")
    print(f"Seed          : {config['seed']}")
    print(f"Output Folder : {config['output'] or 'auto (results/)'}")
    print(f"Layout        : {config['layout']}")

def check_collisions(out_manager: OutputManager):
    collision = out_manager.detect_collision()
    if not collision.exists:
        return True # Safe to proceed
        
    if collision.has_lock and collision.pid:
        import psutil
        is_running = False
        try:
            if psutil.pid_exists(collision.pid):
                is_running = True
        except Exception:
            pass
        if not is_running:
            ans = safe_input(t('stale_lock'))
            if ans.lower() == 'y':
                return True # We will overwrite
            return False
            
    if collision.status == "running":
        ans = safe_input(t('interrupted'))
        if ans == '1': return True
        if ans == '2':
            import shutil
            archive_dir = out_manager.base_path / f"_archive_{int(time.time())}"
            target_dir = out_manager._resolve_target_dir()
            shutil.move(str(target_dir), str(archive_dir))
            return True
        if ans == '3':
            OutputManager.open_folder(str(out_manager._resolve_target_dir()))
            return False
        return False
        
    if collision.has_manifest:
        ans = safe_input(t('collision'))
        if ans == '1': return True
        return False
        
    return True

def run_execution():
    from src.utils.output_manager import OutputLayout
    layout_map = {
        "default": OutputLayout.DEFAULT,
        "date": OutputLayout.DATE,
        "config": OutputLayout.CONFIG,
        "date_config": OutputLayout.DATE_CONFIG,
        "flat": OutputLayout.FLAT
    }
    
    base_out_dir = config['output'] if config['output'] else "results"
    
    def execute_mode(m, algs):
        cfg_data = dict(config)
        cfg_data['mode'] = m
        cfg_data['algorithms'] = algs
        
        out_manager = OutputManager(
            base_path=base_out_dir,
            layout=layout_map.get(config['layout'], OutputLayout.DATE_CONFIG),
            config_data=cfg_data,
            launcher_profile=config.get('profile_name', 'default')
        )
        
        if not check_collisions(out_manager):
            print("Aborted.")
            return
            
        print(f"\nStarting execution for {m} mode...")
        try:
            cmd = [sys.executable, "src/run.py", "--mode", m, "--dataset", config['dataset'],
                   "--trials", str(config['trials']), "--fes", str(config['fes']),
                   "--workers", str(config['workers']), "--device", config['device'],
                   "--scenario", config['scenario'],
                   "--algorithm", ",".join(algs),
                   "--layout", config['layout'],
                   "--launcher-profile", config.get('profile_name', 'default')]
            if config['seed']:
                cmd.extend(["--seed", str(config['seed'])])
            if config['output']:
                cmd.extend(["--output", config['output']])
                
            subprocess.run(cmd, check=True)
            
            # Print folder path instead of blocking with a prompt
            latest_dir = out_manager.get_latest_dir()
            print(f"\nRun completed! Output saved to: {latest_dir}")
                
        except KeyboardInterrupt:
            print("\nExperiment cancelled by user.")
        except Exception as e:
            print(f"\nExperiment failed: {e}")

    if config['mode'] == 'both':
        if config['algorithms_continuous']:
            execute_mode('continuous', config['algorithms_continuous'])
        if config['algorithms_discrete']:
            execute_mode('discrete', config['algorithms_discrete'])
    else:
        execute_mode(config['mode'], config['algorithms'])

def print_dashboard():
    try:
        import torch
        torch_ver = torch.__version__
        cuda_av = "Yes" if torch.cuda.is_available() else "No"
    except ImportError:
        torch_ver = "N/A"
        cuda_av = "N/A"
        
    data_dir = Path("data")
    if data_dir.exists():
        all_csvs = list(data_dir.rglob("*.csv"))
        num_data = len([p for p in all_csvs if p.name not in ('seed_puuids.csv', 'vn2_champion_rates.csv')])
    else:
        num_data = 0
    cpu_count = multiprocessing.cpu_count()
    rec_workers = min(6, max(1, cpu_count - 2))
    
    print("\n--- System Dashboard ---")
    print(f"Python : {platform.python_version()}")
    try:
        import subprocess
        model_name = subprocess.check_output('wmic csproduct get name', shell=True).decode('utf-8').strip().split('\n')[-1].strip()
        if model_name:
            print(f"Machine: {platform.node()} ({model_name})")
        else:
            print(f"Machine: {platform.node()}")
    except Exception:
        print(f"Machine: {platform.node()}")
    
    print(f"CPU    : {platform.processor()} ({cpu_count} cores)")
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        print(f"RAM    : {ram_gb} GB")
    except ImportError:
        pass
    print(f"Workers: Recommended {rec_workers}")
    print(f"Torch  : {torch_ver} (CUDA: {cuda_av})")
    print(f"Dataset: {num_data} available CSVs")
    print("------------------------\n")

def main():
    global LANG
    print(TEXT['en']['title'])
    lang = input(TEXT['en']['lang_prompt']).strip()
    if lang == '2': LANG = 'vi'
    
    print_dashboard()
    
    while True:
        render_state()
        print(t('menu'))
        ans = safe_input("")
        if ans.lower() == 'h': print(t('help'))
        elif ans.lower() == 's':
            if not is_ready():
                print(t('status_incomplete'))
            else:
                run_execution()
        elif ans == '1': select_dataset()
        elif ans == '2': select_mode()
        elif ans == '3': select_scenario()
        elif ans == '4': select_algorithms()
        elif ans == '5': select_integer('trials', 'trials_prompt', 30)
        elif ans == '6': select_integer('fes', 'fes_prompt', 50000)
        elif ans == '7': select_integer('workers', 'workers_prompt', 6)
        elif ans == '8': select_device()
        elif ans == '9': select_integer('seed', 'seed_prompt', 2026)
        elif ans == '10': select_output()
        elif ans == '11': select_layout()
        elif ans == '12': load_config()
        elif ans == '13': save_config()
        elif ans == '14': export_config()
        elif ans == '15': import_config()
        elif ans == '16': delete_config()
        elif ans == '0':
            render_state()
            if not is_ready():
                print(t('status_incomplete'))
                continue
            act = safe_input(t('review_start'))
            if act.lower() == 's':
                run_execution()
            elif act.lower() == 'e':
                continue
            elif act.lower() == 'q':
                break
        else:
            print(t('invalid'))

if __name__ == '__main__':
    main()
