from src.fitness.fitness import calculate_fitness

def test_calculate_fitness():
    # Mock win rates matrix with proper classes
    from src.fitness.fitness import TYPES
    win_rate_matrix = {t1: {t2: 0.5 for t2 in TYPES} for t1 in TYPES}
    avg_duration = 30.0
    
    score, rbi, entropy_pct = calculate_fitness(win_rate_matrix, avg_duration)
    
    assert score is not None
    assert rbi >= 0.0
    assert entropy_pct > 0.0
