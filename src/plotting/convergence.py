import os
from PIL import Image, ImageDraw, ImageFont


def plot_convergence(
    chart_path,
    hx_ga,
    hy_ga,
    hx_pso,
    hy_pso,
    hx_aqea,
    hy_aqea,
    hx_nsga=None,
    hy_nsga=None,
    hx_bal=None,
    hy_bal=None,
):
    print("\n[System] Drawing convergence curve...")
    series = [
        ("GA", hx_ga, hy_ga, "#2563EB"),
        ("PSO", hx_pso, hy_pso, "#F59E0B"),
        ("AQEA", hx_aqea, hy_aqea, "#DC2626"),
    ]
    if hx_nsga:
        series.append(("NSGA-II", hx_nsga, hy_nsga, "#059669"))
    if hx_bal:
        series.append(("Balanced-QEA", hx_bal, hy_bal, "#DB2777"))
    if chart_path.lower().endswith(".png"):
        write_convergence_png(chart_path, series)
    else:
        write_convergence_svg(chart_path, series)
    print(f"[System] Saved {chart_path}")


def write_convergence_png(chart_path, series):
    directory = os.path.dirname(chart_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    width, height = 1000, 600
    left, right, top, bottom = 82, 34, 54, 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(18, bold=False)
    label_font = get_font(15)
    tick_font = get_font(12)
    legend_font = get_font(13)

    all_x = [x for _, xs, _, _ in series for x in xs]
    all_y = [y for _, _, ys, _ in series for y in ys]
    max_x = max(all_x) if all_x else 1
    min_y = min(all_y) if all_y else 0
    max_y = max(all_y) if all_y else 1
    padding = (max_y - min_y) * 0.08 if max_y > min_y else 1
    min_y -= padding
    max_y += padding

    def sx(x):
        return left + (x / max_x) * plot_w

    def sy(y):
        return top + (1 - (y - min_y) / (max_y - min_y)) * plot_h

    draw.text(
        (width / 2 - 250, 18),
        "Convergence Curve: Đánh giá Công bằng dựa trên FEs",
        fill="black",
        font=title_font,
    )

    for tick in range(6):
        x = left + tick * plot_w / 5
        draw.line((x, top, x, top + plot_h), fill="#B8B8B8", width=1)
        value = max_x * tick / 5
        draw.text(
            (x - 18, top + plot_h + 12), f"{value:.0f}", fill="black", font=tick_font
        )
    for tick in range(6):
        y = top + tick * plot_h / 5
        draw.line((left, y, left + plot_w, y), fill="#B8B8B8", width=1)
        value = max_y - (max_y - min_y) * tick / 5
        draw.text((left - 58, y - 8), f"{value:.2f}", fill="black", font=tick_font)

    draw.rectangle((left, top, left + plot_w, top + plot_h), outline="black", width=1)
    draw.text(
        (width / 2 - 190, height - 34),
        "Số lần Đánh giá Hàm (Function Evaluations - FEs)",
        fill="black",
        font=label_font,
    )
    draw.text(
        (16, height / 2 + 120),
        "Điểm số Thích nghi (Multi-objective Fitness)",
        fill="black",
        font=label_font,
        anchor="ls",
    )

    styles = {
        "GA": (6, (8, 5)),
        "PSO": (4, (10, 4, 2, 4)),
        "AQEA": (5, None),
    }
    for name, xs, ys, color in series:
        points = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
        width_line, dash = styles.get(name, (4, None))
        draw_polyline(draw, points, color, width_line, dash)

    legend_x, legend_y = width - 285, height - 135
    draw.rounded_rectangle(
        (legend_x, legend_y, width - 28, height - 30),
        radius=4,
        fill="white",
        outline="#D1D5DB",
    )
    legend_items = [
        ("GA", "Thuật toán Di truyền (GA)", "#2563EB", (8, 5)),
        ("PSO", "Tối ưu Bầy đàn (PSO)", "#F59E0B", (10, 4, 2, 4)),
        ("AQEA", "Lượng tử Thích nghi (AQEA)", "#DC2626", None),
    ]
    for i, (_, label, color, dash) in enumerate(legend_items):
        y = legend_y + 24 + i * 28
        draw_polyline(draw, [(legend_x + 18, y), (legend_x + 55, y)], color, 3, dash)
        draw.text((legend_x + 66, y - 9), label, fill="black", font=legend_font)

    image.save(chart_path)


def get_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_polyline(draw, points, color, width, dash=None):
    if len(points) < 2:
        return
    if dash is None:
        draw.line(points, fill=color, width=width, joint="curve")
        return

    for start, end in zip(points, points[1:]):
        draw_dashed_segment(draw, start, end, color, width, dash)


def draw_dashed_segment(draw, start, end, color, width, dash):
    import math

    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    distance = 0
    dash_index = 0
    draw_on = True
    while distance < length:
        segment = dash[dash_index % len(dash)]
        next_distance = min(length, distance + segment)
        if draw_on:
            a = (x1 + dx * distance, y1 + dy * distance)
            b = (x1 + dx * next_distance, y1 + dy * next_distance)
            draw.line((a, b), fill=color, width=width)
        distance = next_distance
        dash_index += 1
        draw_on = not draw_on


def write_convergence_svg(chart_path, series):
    width, height = 920, 560
    left, right, top, bottom = 84, 32, 64, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_x = [x for _, xs, _, _ in series for x in xs]
    all_y = [y for _, _, ys, _ in series for y in ys]
    max_x = max(all_x) if all_x else 1
    min_y = min(all_y) if all_y else 0
    max_y = max(all_y) if all_y else 1
    if max_y == min_y:
        max_y += 1

    def sx(x):
        return left + (x / max_x) * plot_w

    def sy(y):
        return top + (1 - (y - min_y) / (max_y - min_y)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.small{font-size:12px;fill:#4B5563}.title{font-size:22px;font-weight:700}</style>",
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="32" y="36" class="title">Convergence Curve: Game Balance Optimization</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9CA3AF"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9CA3AF"/>',
        f'<text x="{left + plot_w / 2 - 70}" y="{height - 24}" class="small">Function Evaluations (FEs)</text>',
        f'<text x="16" y="{top + plot_h / 2}" class="small" transform="rotate(-90 16 {top + plot_h / 2})">Fitness</text>',
    ]

    for tick in range(6):
        x = left + tick * plot_w / 5
        value = max_x * tick / 5
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#F3F4F6"/>'
        )
        parts.append(
            f'<text x="{x - 12:.2f}" y="{top + plot_h + 18}" class="small">{value:.0f}</text>'
        )

    for tick in range(5):
        y = top + tick * plot_h / 4
        value = max_y - (max_y - min_y) * tick / 4
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#F3F4F6"/>'
        )
        parts.append(
            f'<text x="{left - 56}" y="{y + 4:.2f}" class="small">{value:.2f}</text>'
        )

    legend_x = left + plot_w - 190
    for index, (name, xs, ys, color) in enumerate(series):
        if not xs or not ys:
            continue
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        ly = top + 18 + index * 22
        parts.append(
            f'<rect x="{legend_x}" y="{ly - 10}" width="14" height="14" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 22}" y="{ly + 2}" class="small">{name}</text>'
        )

    parts.append("</svg>")
    with open(chart_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
