import sys
import time

_progress_start_times = {}

def print_progress_bar(algorithm_name, current_fe, max_fe, best_score, length=40):
    global _progress_start_times
    
    # Initialize start time if it's the first call for this algorithm/trial
    if algorithm_name not in _progress_start_times:
        _progress_start_times[algorithm_name] = time.time()
        
    start_time = _progress_start_times[algorithm_name]
    elapsed = time.time() - start_time
    
    percent = current_fe / float(max_fe) if max_fe > 0 else 0
    filled_length = int(length * percent)
    bar = '=' * filled_length + '-' * (length - filled_length)
    
    # Calculate ETA
    if percent > 0:
        total_estimated = elapsed / percent
        eta_seconds = max(0, total_estimated - elapsed)
        eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))
    else:
        eta_str = "00:00"
        
    score_str = f"{best_score:.4f}" if isinstance(best_score, (float, int)) else str(best_score)
    
    sys.stdout.write(f'\r[{algorithm_name}] [{bar}] {current_fe}/{max_fe} FEs | Best: {score_str} | ETA: {eta_str}      ')
    sys.stdout.flush()
    
    # Cleanup when finished
    if current_fe >= max_fe:
        sys.stdout.write('\n')
        sys.stdout.flush()
        if algorithm_name in _progress_start_times:
            del _progress_start_times[algorithm_name]
