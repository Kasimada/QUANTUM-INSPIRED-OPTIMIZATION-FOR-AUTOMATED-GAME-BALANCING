# Nguồn dữ liệu: Thiết lập dựa trên triết lý Thiết kế Game của Liên Minh Huyền Thoại (League of Legends).
# TYPE_CHART là ma trận heuristic proxy đại diện cho cơ chế khắc chế Class của LMHT,
# thay thế cho cơ chế kỹ năng (abilities) không thể mô phỏng trực tiếp trong môi trường 1vs1.
# Giá trị > 1.0: Lớp tấn công có lợi thế đối kháng (counter advantage).
# Giá trị < 1.0: Lớp tấn công ở thế bất lợi (counter-picked).
# Căn cứ theo: Riot Games Class Design Philosophy & metagame counter relationships.

TYPES = ["Fighter", "Mage", "Assassin", "Marksman", "Tank", "Support"]

# Hệ số nhân sát thương theo lớp nhân vật (Class-based Damage Multiplier)
# Hàng = Lớp tấn công (Attacker), Cột = Lớp bị tấn công (Defender)
TYPE_CHART = {
    # Sát Thủ: Khắc chế Xạ Thủ/Pháp Sư/Hỗ Trợ, yếu trước Đỡ Đòn/Chiến Binh
    "Assassin": {
        "Assassin": 1.00,
        "Marksman": 1.30,
        "Mage": 1.20,
        "Fighter": 0.85,
        "Tank": 0.70,
        "Support": 1.40,
    },
    # Xạ Thủ: Xuyên giáp Tank, yếu trước Sát Thủ (không có kỹ năng thoát)
    "Marksman": {
        "Assassin": 0.75,
        "Marksman": 1.00,
        "Mage": 1.00,
        "Fighter": 0.90,
        "Tank": 1.35,
        "Support": 1.15,
    },
    # Pháp Sư: Sát thương phép xuyên qua giáp của Chiến Binh/Đỡ Đòn
    "Mage": {
        "Assassin": 0.80,
        "Marksman": 1.00,
        "Mage": 1.00,
        "Fighter": 1.25,
        "Tank": 1.10,
        "Support": 1.15,
    },
    # Chiến Binh: Áp chế Sát Thủ bằng độ bền, yếu trước Pháp Sư (kháng phép thấp)
    "Fighter": {
        "Assassin": 1.20,
        "Marksman": 1.10,
        "Mage": 0.85,
        "Fighter": 1.00,
        "Tank": 1.15,
        "Support": 1.30,
    },
    # Đỡ Đòn: Cứng trước Sát Thủ, yếu trước Xạ Thủ (máu bị bào mòn từ xa)
    "Tank": {
        "Assassin": 1.30,
        "Marksman": 0.70,
        "Mage": 0.90,
        "Fighter": 0.85,
        "Tank": 1.00,
        "Support": 1.10,
    },
    # Hỗ Trợ: Sát thương thấp nhưng khả năng sinh tồn bù qua hồi máu (proxy utility)
    "Support": {
        "Assassin": 0.65,
        "Marksman": 0.80,
        "Mage": 0.80,
        "Fighter": 0.70,
        "Tank": 0.90,
        "Support": 1.00,
    },
}

# Ma trận mục tiêu Heuristic (T_ij) — Target Metagame Matrix
# Xác định tỷ lệ thắng kỳ vọng của Class hàng ngang khi đối đầu Class cột dọc.
# 0.65: Khắc chế cứng (Hard Counter)
# 0.60: Khắc chế mềm (Soft Counter)
# 0.50: Cân bằng / Kèo kỹ năng (Skill Matchup)
# 0.30-0.40: Bất lợi rõ ràng (Counter-picked)
TARGET_MATRIX = {
    "Assassin": {
        "Assassin": 0.50,
        "Marksman": 0.65,
        "Mage": 0.60,
        "Fighter": 0.40,
        "Tank": 0.35,
        "Support": 0.70,
    },
    "Marksman": {
        "Assassin": 0.35,
        "Marksman": 0.50,
        "Mage": 0.50,
        "Fighter": 0.45,
        "Tank": 0.65,
        "Support": 0.60,
    },
    "Mage": {
        "Assassin": 0.40,
        "Marksman": 0.50,
        "Mage": 0.50,
        "Fighter": 0.60,
        "Tank": 0.55,
        "Support": 0.60,
    },
    "Fighter": {
        "Assassin": 0.60,
        "Marksman": 0.55,
        "Mage": 0.40,
        "Fighter": 0.50,
        "Tank": 0.60,
        "Support": 0.65,
    },
    "Tank": {
        "Assassin": 0.65,
        "Marksman": 0.35,
        "Mage": 0.45,
        "Fighter": 0.40,
        "Tank": 0.50,
        "Support": 0.55,
    },
    "Support": {
        "Assassin": 0.30,
        "Marksman": 0.40,
        "Mage": 0.40,
        "Fighter": 0.35,
        "Tank": 0.45,
        "Support": 0.50,
    },
}

ASYMMETRIC_TARGET_MATRIX = {
    "Assassin": {
        "Assassin": 0.50,
        "Marksman": 0.65,
        "Mage": 0.60,
        "Fighter": 0.40,
        "Tank": 0.35,
        "Support": 0.70,
    },
    "Marksman": {
        "Assassin": 0.35,
        "Marksman": 0.50,
        "Mage": 0.50,
        "Fighter": 0.45,
        "Tank": 0.65,
        "Support": 0.60,
    },
    "Mage": {
        "Assassin": 0.40,
        "Marksman": 0.50,
        "Mage": 0.50,
        "Fighter": 0.60,
        "Tank": 0.55,
        "Support": 0.60,
    },
    "Fighter": {
        "Assassin": 0.60,
        "Marksman": 0.55,
        "Mage": 0.40,
        "Fighter": 0.50,
        "Tank": 0.60,
        "Support": 0.65,
    },
    "Tank": {
        "Assassin": 0.65,
        "Marksman": 0.35,
        "Mage": 0.45,
        "Fighter": 0.40,
        "Tank": 0.50,
        "Support": 0.55,
    },
    "Support": {
        "Assassin": 0.30,
        "Marksman": 0.40,
        "Mage": 0.40,
        "Fighter": 0.35,
        "Tank": 0.45,
        # Handicap: Support win rate reduced intentionally
        "Support": 0.40, 
    },
}

def get_target_matrix(scenario="symmetric"):
    if scenario == "asymmetric_support_handicap":
        return ASYMMETRIC_TARGET_MATRIX
    return TARGET_MATRIX

def get_supported_scenarios() -> list:
    return ["symmetric", "asymmetric_support_handicap"]
