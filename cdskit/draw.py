import sys
from html import escape
from math import ceil

from cdskit.codonutil import (
    ambiguous_codon_counts,
    codon_has_missing,
    codon_is_ambiguous,
    codon_is_stop,
)
from cdskit.trimcodon import choose_kept_codon_sites, summarize_codon_site, validate_fraction
from cdskit.util import (
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_aligned,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
)


DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 720
DEFAULT_ROW_HEIGHT = 24
DEFAULT_LABEL_WIDTH = 180
DEFAULT_TITLE = 'CDSKIT draw'
DEFAULT_TOP_N = 10

MISSING_COLOR = '#d9d9d9'
COMPLETE_COLOR = '#4e79a7'
AMBIGUOUS_COLOR = '#f28e2b'
STOP_COLOR = '#e15759'
KEEP_COLOR = '#59a14f'
REMOVE_COLOR = '#bab0ab'
BAR_COLOR = '#8f63a8'


def positive_int_arg(name, value, default):
    if value is None:
        value = default
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise Exception(f'{name} must be an integer. Exiting.\n')
    if value <= 0:
        raise Exception(f'{name} must be greater than zero. Exiting.\n')
    return value


def nonnegative_int_arg(name, value, default):
    if value is None:
        value = default
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise Exception(f'{name} must be an integer. Exiting.\n')
    if value < 0:
        raise Exception(f'{name} must be greater than or equal to zero. Exiting.\n')
    return value


def fmt_num(value):
    if isinstance(value, int):
        return str(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f'{value:.2f}'.rstrip('0').rstrip('.')


def svg_text(text, x, y, size=12, anchor='start', weight='normal', cls=None):
    class_attr = f' class="{cls}"' if cls else ''
    return (
        f'<text{class_attr} x="{fmt_num(x)}" y="{fmt_num(y)}" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">'
        f'{escape(str(text))}</text>'
    )


def svg_rect(x, y, width, height, cls, extra=''):
    extra_attr = f' {extra}' if extra else ''
    return (
        f'<rect class="{cls}" x="{fmt_num(x)}" y="{fmt_num(y)}" '
        f'width="{fmt_num(width)}" height="{fmt_num(height)}"{extra_attr} />'
    )


def svg_line(x1, y1, x2, y2, cls, extra=''):
    extra_attr = f' {extra}' if extra else ''
    return (
        f'<line class="{cls}" x1="{fmt_num(x1)}" y1="{fmt_num(y1)}" '
        f'x2="{fmt_num(x2)}" y2="{fmt_num(y2)}"{extra_attr} />'
    )


def truncate_label(label, max_chars):
    label = str(label)
    if max_chars <= 0 or len(label) <= max_chars:
        return label
    if max_chars <= 3:
        return label[:max_chars]
    return label[: max_chars - 3] + '...'


def classify_codon(codon, codontable):
    if codon_has_missing(codon):
        return 'missing'
    if codon_is_stop(codon=codon, codontable=codontable):
        return 'stop'
    if codon_is_ambiguous(codon):
        return 'ambiguous'
    return 'complete'


def summarize_draw(records, codontable, min_occupancy, max_ambiguous_fraction, drop_stop_codon):
    seq_strings = [str(record.seq) for record in records]
    if len(records) == 0:
        return {
            'site_summaries': list(),
            'kept_sites': list(),
            'sequence_ambiguous_counts': list(),
            'ambiguous_total': 0,
            'evaluable_total': 0,
        }

    num_sites = len(seq_strings[0]) // 3
    site_summaries = [
        summarize_codon_site(seq_strings=seq_strings, codon_site=codon_site, codontable=codontable)
        for codon_site in range(num_sites)
    ]
    kept_sites = choose_kept_codon_sites(
        site_summaries=site_summaries,
        num_sequences=len(records),
        min_occupancy=min_occupancy,
        max_ambiguous_fraction=max_ambiguous_fraction,
        drop_stop_codon=drop_stop_codon,
    )

    sequence_ambiguous_counts = list()
    ambiguous_total = 0
    evaluable_total = 0
    for record in records:
        ambiguous, evaluable = ambiguous_codon_counts(str(record.seq))
        sequence_ambiguous_counts.append((record.id, ambiguous))
        ambiguous_total += ambiguous
        evaluable_total += evaluable
    sequence_ambiguous_counts.sort(key=lambda item: (-item[1], item[0]))

    return {
        'site_summaries': site_summaries,
        'kept_sites': kept_sites,
        'sequence_ambiguous_counts': sequence_ambiguous_counts,
        'ambiguous_total': ambiguous_total,
        'evaluable_total': evaluable_total,
    }


def build_svg(records, args, summary):
    width = float(positive_int_arg('--width', getattr(args, 'width', DEFAULT_WIDTH), DEFAULT_WIDTH))
    height = float(positive_int_arg('--height', getattr(args, 'height', DEFAULT_HEIGHT), DEFAULT_HEIGHT))
    row_height = float(positive_int_arg('--row_height', getattr(args, 'row_height', DEFAULT_ROW_HEIGHT), DEFAULT_ROW_HEIGHT))
    label_width = float(positive_int_arg('--label_width', getattr(args, 'label_width', DEFAULT_LABEL_WIDTH), DEFAULT_LABEL_WIDTH))
    title = getattr(args, 'title', DEFAULT_TITLE) or DEFAULT_TITLE
    top_n = nonnegative_int_arg('--top_n', getattr(args, 'top_n', DEFAULT_TOP_N), DEFAULT_TOP_N)

    margin_left = 18.0
    margin_right = 18.0
    margin_top = 18.0
    margin_bottom = 18.0
    title_h = 28.0 if title else 0.0
    summary_h = 22.0
    legend_h = 28.0
    strip_h = 6.0
    gap = 10.0

    num_records = len(records)
    num_sites = len(summary['site_summaries'])

    include_side_chart = (
        top_n > 0
        and num_records > 0
        and width >= (label_width + 420.0)
        and len(summary['sequence_ambiguous_counts']) > 0
    )
    side_width = 250.0 if include_side_chart else 0.0
    side_gap = 22.0 if include_side_chart else 0.0

    map_x0 = margin_left + label_width
    map_x1 = width - margin_right - side_width - side_gap
    map_width = max(1.0, map_x1 - map_x0)
    tile_w = map_width / num_sites if num_sites > 0 else map_width

    grid_top = margin_top + title_h + summary_h + legend_h + gap
    grid_height = num_records * row_height
    axis_top = grid_top + strip_h + grid_height + 10.0
    axis_h = 24.0
    required_height = axis_top + axis_h + margin_bottom
    canvas_height = max(height, required_height)

    chars_per_label = max(4, int((label_width - 12.0) / 7.0))
    tick_step = 1
    if num_sites > 0:
        tick_step = max(1, int(ceil(num_sites / max(1.0, map_width / 80.0))))

    out = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{fmt_num(width)}" '
        f'height="{fmt_num(canvas_height)}" viewBox="0 0 {fmt_num(width)} {fmt_num(canvas_height)}">'
    )
    out.append(
        '<defs>'
        '<style>'
        'text { font-family: DejaVu Sans, Arial, sans-serif; fill: #1f1f1f; }'
        '.title { font-size: 20px; font-weight: 700; }'
        '.summary { font-size: 12px; fill: #444; }'
        '.legend { font-size: 12px; fill: #333; }'
        '.seq-label { font-size: 11px; fill: #222; }'
        '.axis { stroke: #555; stroke-width: 1; }'
        '.tick { stroke: #777; stroke-width: 1; }'
        '.tick-label { font-size: 10px; fill: #555; }'
        '.tile { stroke: #ffffff; stroke-width: 0.5; shape-rendering: crispEdges; }'
        '.tile.complete { fill: ' + COMPLETE_COLOR + '; }'
        '.tile.missing { fill: ' + MISSING_COLOR + '; }'
        '.tile.ambiguous { fill: ' + AMBIGUOUS_COLOR + '; }'
        '.tile.stop { fill: ' + STOP_COLOR + '; }'
        '.site-strip.keep { fill: ' + KEEP_COLOR + '; }'
        '.site-strip.remove { fill: ' + REMOVE_COLOR + '; }'
        '.legend-swatch.complete { fill: ' + COMPLETE_COLOR + '; }'
        '.legend-swatch.missing { fill: ' + MISSING_COLOR + '; }'
        '.legend-swatch.ambiguous { fill: ' + AMBIGUOUS_COLOR + '; }'
        '.legend-swatch.stop { fill: ' + STOP_COLOR + '; }'
        '.legend-swatch.keep { fill: ' + KEEP_COLOR + '; }'
        '.legend-swatch.remove { fill: ' + REMOVE_COLOR + '; }'
        '.bar { fill: ' + BAR_COLOR + '; }'
        '.bar-label { font-size: 10px; fill: #333; }'
        '.bar-value { font-size: 10px; fill: #333; }'
        '</style>'
        '</defs>'
    )

    out.append(svg_text(title, margin_left, margin_top + 16, size=20, weight='700', cls='title'))
    out.append(
        svg_text(
            f"Sequences: {len(records)} | Codon sites: {num_sites} | "
            f"Kept sites: {len(summary['kept_sites'])} | Removed sites: {num_sites - len(summary['kept_sites'])}",
            margin_left,
            margin_top + 38,
            size=12,
            cls='summary',
        )
    )
    out.append(
        svg_text(
            f"Thresholds: occupancy >= {getattr(args, 'min_occupancy', 0):.2f}, "
            f"ambiguous fraction <= {getattr(args, 'max_ambiguous_fraction', 1):.2f}, "
            f"drop stop codon = {bool(getattr(args, 'drop_stop_codon', False))}",
            margin_left,
            margin_top + 56,
            size=12,
            cls='summary',
        )
    )

    legend_y = margin_top + title_h + 14.0
    legend_x = margin_left
    legend_items = [
        ('complete', 'complete'),
        ('missing', 'missing'),
        ('ambiguous', 'ambiguous'),
        ('stop', 'stop'),
        ('keep', 'keep'),
        ('remove', 'remove'),
    ]
    for idx, (cls, label) in enumerate(legend_items):
        x = legend_x + idx * 118.0
        out.append(svg_rect(x, legend_y, 12, 12, f'legend-swatch {cls}'))
        out.append(svg_text(label, x + 18, legend_y + 11, size=12, cls='legend'))

    if num_records == 0 or num_sites == 0:
        out.append(svg_text('No sequences to draw', margin_left, grid_top + 20, size=13, cls='summary'))
    else:
        for site_idx, site_summary in enumerate(summary['site_summaries']):
            x = map_x0 + site_idx * tile_w
            state = 'keep' if site_idx in summary['kept_sites'] else 'remove'
            out.append(svg_rect(x, grid_top, tile_w, strip_h, f'site-strip {state}'))
            if site_idx == 0 or (site_idx + 1) % tick_step == 0 or site_idx == num_sites - 1:
                tick_x = x + tile_w / 2.0
                out.append(svg_line(tick_x, grid_top + strip_h + grid_height + 3, tick_x, grid_top + strip_h + grid_height + 8, 'tick'))
                out.append(
                    svg_text(
                        site_idx + 1,
                        tick_x,
                        grid_top + strip_h + grid_height + 18,
                        size=10,
                        anchor='middle',
                        cls='tick-label',
                    )
                )

            for row_idx, record in enumerate(records):
                seq = str(record.seq)
                codon = seq[site_idx * 3 : site_idx * 3 + 3]
                category = classify_codon(codon=codon, codontable=args.codontable)
                tile_y = grid_top + strip_h + row_idx * row_height + 1.0
                tile_h = max(1.0, row_height - 2.0)
                out.append(
                    svg_rect(
                        x,
                        tile_y,
                        tile_w,
                        tile_h,
                        f'tile {category}',
                        extra=f'data-seq="{escape(record.id)}" data-site="{site_idx + 1}"',
                    )
                )

        out.append(svg_line(map_x0, grid_top, map_x0 + map_width, grid_top, 'axis'))
        out.append(svg_line(map_x0, grid_top + strip_h + grid_height, map_x0 + map_width, grid_top + strip_h + grid_height, 'axis'))

        for row_idx, record in enumerate(records):
            label_y = grid_top + strip_h + row_idx * row_height + row_height / 2.0 + 4.0
            out.append(
                svg_text(
                    truncate_label(record.id, chars_per_label),
                    map_x0 - 6,
                    label_y,
                    size=11,
                    anchor='end',
                    cls='seq-label',
                )
            )

        if include_side_chart:
            side_x0 = map_x1 + side_gap
            side_y0 = grid_top
            chart_title = 'Top ambiguous codon counts'
            out.append(svg_text(chart_title, side_x0, side_y0 - 6, size=13, weight='700', cls='legend'))
            top_items = summary['sequence_ambiguous_counts'][:top_n]
            max_count = max((count for _, count in top_items), default=0)
            bar_width = max(1.0, side_width - 92.0)
            bar_h = max(12.0, min(20.0, row_height - 4.0))
            for idx, (seq_id, count) in enumerate(top_items):
                y = side_y0 + idx * (bar_h + 6.0)
                out.append(svg_text(truncate_label(seq_id, 18), side_x0, y + bar_h - 2, size=10, cls='bar-label'))
                w = 0.0 if max_count == 0 else bar_width * (count / max_count)
                out.append(svg_rect(side_x0 + 80.0, y, w, bar_h, 'bar'))
                out.append(svg_text(count, side_x0 + 84.0 + max(w, 12.0), y + bar_h - 2, size=10, cls='bar-value'))

    out.append('</svg>')
    return '\n'.join(out)


def write_svg(svg_text_content, outfile):
    if outfile == '-':
        sys.stdout.write(svg_text_content)
        if not svg_text_content.endswith('\n'):
            sys.stdout.write('\n')
        return
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(svg_text_content)
        if not svg_text_content.endswith('\n'):
            f.write('\n')


def draw_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    _ = resolve_threads(getattr(args, 'threads', 1))
    stop_if_not_dna(records=records, label='--seqfile')
    stop_if_not_aligned(records=records)
    stop_if_not_multiple_of_three(records=records)
    stop_if_invalid_codontable(args.codontable)
    min_occupancy = validate_fraction(name='--min_occupancy', value=getattr(args, 'min_occupancy', 0.5))
    max_ambiguous_fraction = validate_fraction(
        name='--max_ambiguous_fraction',
        value=getattr(args, 'max_ambiguous_fraction', 1.0),
    )
    drop_stop_codon = bool(getattr(args, 'drop_stop_codon', False))
    summary = summarize_draw(
        records=records,
        codontable=args.codontable,
        min_occupancy=min_occupancy,
        max_ambiguous_fraction=max_ambiguous_fraction,
        drop_stop_codon=drop_stop_codon,
    )
    svg = build_svg(records=records, args=args, summary=summary)
    write_svg(svg_text_content=svg, outfile=getattr(args, 'outfile', '-'))
