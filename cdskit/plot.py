import copy
import io
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'

from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from cdskit.codonutil import codon_has_missing, get_forward_table, get_stop_codons
from cdskit.draw import classify_codon, summarize_draw
from cdskit.trimcodon import choose_kept_codon_sites, summarize_codon_site, validate_fraction
from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_aligned,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
)
from cdskit.validate import sequence_ambiguous_codon_counts


DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 720
DEFAULT_ROW_HEIGHT = 18
DEFAULT_LABEL_WIDTH = 180
DEFAULT_WRAP = 60
VALID_PLOT_MODES = frozenset(('summary', 'map', 'msa'))
VALID_PLOT_FORMATS = frozenset(('pdf', 'svg', 'png'))

COL_OCCUPANCY = '#2e7d32'
COL_AMBIGUOUS = '#ef6c00'
COL_THRESHOLD = '#546e7a'
COL_STOP = '#c62828'
COL_KEEP = '#c8e6c9'
COL_REMOVE = '#d7dde5'
COL_BAR = '#ffb74d'
COL_BAR_EDGE = '#ef6c00'
COL_COMPLETE = '#4e79a7'
COL_MISSING = '#d9d9d9'
COL_CONSERVATION = '#546e7a'
COL_CONSERVATION_BG = '#eef2f5'
COL_A = '#8ddf6f'
COL_C = '#72c7ff'
COL_G = '#ffd966'
COL_T = '#ff99c8'
COL_N = '#d5d9df'
COL_OTHER = '#d9c2ff'
COL_GAP = '#ffffff'
COL_MSA_BORDER = '#d7dde5'
COL_MSA_AA_BG = '#f3f6f8'
COL_MSA_LOGO_BG = '#f7f9fb'
COL_MSA_ROW_BG = '#fbfcfe'
COL_MSA_TEXT = '#263238'
COL_MSA_SUBTEXT = '#607d8b'

LOGO_FONT = FontProperties(family='DejaVu Sans', weight='bold')
NT_FONT_FAMILY = 'DejaVu Sans Mono'

STATE_TO_INT = {
    'complete': 0,
    'missing': 1,
    'ambiguous': 2,
    'stop': 3,
}


def _int_arg(name, value, default):
    if value is None:
        value = default
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise Exception(f'{name} must be an integer. Exiting.\n')
    if value <= 0:
        raise Exception(f'{name} must be greater than zero. Exiting.\n')
    return value


def _nonnegative_int_arg(name, value, default):
    if value is None:
        value = default
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise Exception(f'{name} must be an integer. Exiting.\n')
    if value < 0:
        raise Exception(f'{name} must be greater than or equal to zero. Exiting.\n')
    return value


def _normalize_mode(mode):
    mode_str = str(mode).lower()
    if mode_str not in VALID_PLOT_MODES:
        valid = ', '.join(sorted(VALID_PLOT_MODES))
        raise Exception(f'Invalid --mode: {mode}. Choose from {valid}. Exiting.\n')
    return mode_str


def _normalize_plotformat(plotformat, outfile):
    plotformat_str = str(plotformat).lower()
    if plotformat_str == 'auto':
        if outfile != '-':
            suffix = Path(outfile).suffix.lower().lstrip('.')
            if suffix in VALID_PLOT_FORMATS:
                return suffix
        return 'pdf'
    if plotformat_str not in VALID_PLOT_FORMATS:
        valid = ', '.join(sorted(VALID_PLOT_FORMATS | {'auto'}))
        raise Exception(f'Invalid --format: {plotformat}. Choose from {valid}. Exiting.\n')
    return plotformat_str


def _fmt_count(value):
    return f'{int(value):,}'


def _truncate_text(text, limit=28):
    text = str(text)
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + '…'


def _compute_sequence_stats(records, threads):
    worker = lambda record: {
        'seq_id': record.id,
        'ambiguous_codons': sequence_ambiguous_codon_counts(str(record.seq))[0],
    }
    return parallel_map_ordered(items=records, worker=worker, threads=threads)


def _build_summary_text(num_sequences, num_sites, kept_sites, removed_sites, min_occupancy, max_ambiguous_fraction, drop_stop_codon):
    stop_txt = 'drop stop codons' if drop_stop_codon else 'keep stop codons'
    return ' | '.join([
        f'{_fmt_count(num_sequences)} sequences',
        f'{_fmt_count(num_sites)} codon sites',
        f'{_fmt_count(kept_sites)} kept',
        f'{_fmt_count(removed_sites)} removed',
        f'occupancy >= {min_occupancy:.2f}',
        f'ambiguous fraction <= {max_ambiguous_fraction:.2f}',
        stop_txt,
    ])


def _make_site_ticks(num_sites):
    if num_sites <= 0:
        return list()
    if num_sites <= 12:
        return list(range(1, num_sites + 1))
    step = max(1, num_sites // 10)
    ticks = list(range(1, num_sites + 1, step))
    if ticks[-1] != num_sites:
        ticks.append(num_sites)
    return ticks


def _build_state_matrix(records, codontable):
    if len(records) == 0:
        return np.zeros((0, 0), dtype=int)
    num_sites = len(records[0].seq) // 3
    matrix = np.zeros((len(records), num_sites), dtype=int)
    for row_idx, record in enumerate(records):
        seq = str(record.seq)
        for site_idx in range(num_sites):
            codon = seq[site_idx * 3:site_idx * 3 + 3]
            matrix[row_idx, site_idx] = STATE_TO_INT[classify_codon(codon=codon, codontable=codontable)]
    return matrix


def _msa_color_for_char(ch):
    ch_upper = str(ch).upper()
    if ch_upper == 'A':
        return COL_A
    if ch_upper == 'C':
        return COL_C
    if ch_upper == 'G':
        return COL_G
    if ch_upper in ('T', 'U'):
        return COL_T
    if ch_upper == 'N':
        return COL_N
    if ch_upper in ('-', '?', '.'):
        return COL_GAP
    return COL_OTHER


def _aa_color_for_char(ch):
    ch_upper = str(ch).upper()
    if ch_upper in ('D', 'E'):
        return '#d73027'
    if ch_upper in ('K', 'R', 'H'):
        return '#4575b4'
    if ch_upper in ('S', 'T', 'N', 'Q'):
        return '#74add1'
    if ch_upper in ('A', 'V', 'I', 'L', 'M'):
        return '#66bd63'
    if ch_upper in ('F', 'W', 'Y'):
        return '#1a9850'
    if ch_upper == 'G':
        return '#fdae61'
    if ch_upper == 'P':
        return '#f46d43'
    if ch_upper == 'C':
        return '#ffd92f'
    if ch_upper == '*':
        return '#212121'
    if ch_upper == '-':
        return '#b0bec5'
    return '#9c89b8'


def _translate_codon_for_msa(codon, codontable, forward_table, stop_codons):
    codon_upper = str(codon).upper()
    if len(codon_upper) != 3:
        return '?'
    if codon_has_missing(codon_upper):
        return '-'
    if any(ch not in 'ACGT' for ch in codon_upper):
        return '?'
    if codon_upper in stop_codons:
        return '*'
    return forward_table.get(codon_upper, '?')


def _consensus_nt_for_codons(codons):
    out = list()
    for pos in range(3):
        chars = [codon[pos].upper() for codon in codons if len(codon) == 3]
        informative = [ch for ch in chars if ch not in ('-', '?', '.')]
        if len(informative) == 0:
            out.append('-')
            continue
        counts = Counter(informative)
        out.append(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0])
    return ''.join(out)


def _build_msa_summary(records, codontable):
    seq_strings = [str(record.seq).upper() for record in records]
    num_codons = len(seq_strings[0]) // 3 if len(seq_strings) > 0 else 0
    forward_table = get_forward_table(codontable=codontable)
    stop_codons = get_stop_codons(codontable=codontable)
    seq_codons = [
        [seq[codon_idx * 3:codon_idx * 3 + 3] for codon_idx in range(num_codons)]
        for seq in seq_strings
    ]
    seq_aas = [
        [
            _translate_codon_for_msa(
                codon=codon,
                codontable=codontable,
                forward_table=forward_table,
                stop_codons=stop_codons,
            )
            for codon in row_codons
        ]
        for row_codons in seq_codons
    ]

    consensus_codons = list()
    consensus_aas = list()
    aa_logo_frequencies = list()

    for codon_idx in range(num_codons):
        codons = [seq[codon_idx * 3:codon_idx * 3 + 3] for seq in seq_strings]
        valid_codons = [
            codon.upper()
            for codon in codons
            if (len(codon) == 3)
            and (not codon_has_missing(codon))
            and all(ch in 'ACGT' for ch in codon.upper())
        ]

        if len(valid_codons) > 0:
            codon_counts = Counter(valid_codons)
            consensus_codon = sorted(codon_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            aa_counts = Counter(
                _translate_codon_for_msa(
                    codon=codon,
                    codontable=codontable,
                    forward_table=forward_table,
                    stop_codons=stop_codons,
                )
                for codon in valid_codons
            )
        else:
            if len(codons) > 0 and all((len(codon) == 3) and all(ch in '-?.' for ch in codon.upper()) for codon in codons):
                consensus_codon = '---'
            else:
                consensus_codon = _consensus_nt_for_codons(codons=codons)
            aa_counts = Counter()

        consensus_codons.append(consensus_codon)
        consensus_aas.append(
            _translate_codon_for_msa(
                codon=consensus_codon,
                codontable=codontable,
                forward_table=forward_table,
                stop_codons=stop_codons,
            )
        )
        aa_total = sum(aa_counts.values())
        if aa_total <= 0:
            aa_logo_frequencies.append(list())
        else:
            aa_logo_frequencies.append([
                (aa, count / aa_total)
                for aa, count in sorted(aa_counts.items(), key=lambda item: (item[1], item[0]))
            ])

    return {
        'seq_strings': seq_strings,
        'seq_codons': seq_codons,
        'seq_aas': seq_aas,
        'consensus_codons': consensus_codons,
        'consensus_aas': consensus_aas,
        'aa_logo_frequencies': aa_logo_frequencies,
    }


def _draw_msa_rect(ax, x, y, width, height, facecolor, edgecolor='#ffffff', linewidth=0.6):
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def _draw_logo_letter(ax, letter, x, y, width, height):
    if height <= 0:
        return
    text_path = TextPath((0, 0), str(letter), size=1, prop=LOGO_FONT)
    bbox = text_path.get_extents()
    if (bbox.width <= 0) or (bbox.height <= 0):
        return
    width_factor = (1.0 / 3.0) if str(letter).upper() == 'I' else 1.0
    usable_width = width * 0.82 * width_factor
    usable_height = height * 0.92
    scale_y = usable_height / bbox.height
    scale_x = usable_width / bbox.width
    draw_width = bbox.width * scale_x
    draw_height = bbox.height * scale_y
    x_left = x + (width - draw_width) / 2.0
    y_top = y + (height - draw_height) / 2.0
    transform = (
        Affine2D()
        .scale(scale_x, -scale_y)
        .translate(x_left - bbox.x0 * scale_x, y_top + bbox.y1 * scale_y)
        + ax.transData
    )
    ax.add_patch(PathPatch(text_path, transform=transform, facecolor=_aa_color_for_char(letter), edgecolor='none'))


def _draw_centered_box_text(
    ax,
    center_x,
    center_y,
    box_width,
    box_height,
    facecolor,
    text,
    fontsize,
    text_color,
    fontfamily=None,
    fontweight='normal',
    linewidth=0.45,
):
    _draw_msa_rect(
        ax,
        center_x - (box_width / 2.0),
        center_y - (box_height / 2.0),
        box_width,
        box_height,
        facecolor,
        edgecolor=COL_MSA_BORDER,
        linewidth=linewidth,
    )
    ax.text(
        center_x,
        center_y,
        text,
        ha='center',
        va='center',
        fontsize=fontsize,
        fontfamily=fontfamily,
        fontweight=fontweight,
        color=text_color,
    )


def _draw_boxed_tick(ax, center_x, center_y, box_width, box_height, text, fontsize):
    _draw_centered_box_text(
        ax=ax,
        center_x=center_x,
        center_y=center_y,
        box_width=box_width,
        box_height=box_height,
        facecolor=COL_MSA_LOGO_BG,
        text=text,
        fontsize=fontsize,
        text_color=COL_MSA_SUBTEXT,
        fontfamily=NT_FONT_FAMILY,
        linewidth=0.5,
    )


def _draw_aa_logo(ax, freqs, nt_start, y_top, logo_height):
    logo_box_width = 2.52
    logo_box_left = nt_start + 1.0 - (logo_box_width / 2.0)
    _draw_msa_rect(ax, logo_box_left, y_top, logo_box_width, logo_height, COL_MSA_LOGO_BG, edgecolor=COL_MSA_BORDER, linewidth=0.6)
    current_bottom = y_top + logo_height
    for aa, freq in freqs:
        letter_height = logo_height * float(freq)
        if letter_height <= 0.02:
            continue
        letter_top = current_bottom - letter_height
        _draw_logo_letter(ax, aa, logo_box_left + 0.10, letter_top, logo_box_width - 0.20, letter_height)
        current_bottom = letter_top


def _draw_sequence_split_row(ax, codon, aa, nt_start, row_top, panel_height, aa_font_size, codon_font_size, aa_alpha=0.16):
    panel_top = row_top
    content_top = panel_top + 0.005
    aa_height = 0.18
    codon_gap = 0.005
    codon_height = 0.055
    aa_center_y = content_top + (aa_height / 2.0)
    codon_center_y = content_top + aa_height + codon_gap + (codon_height / 2.0)

    _draw_msa_rect(
        ax,
        nt_start - 0.5,
        panel_top,
        3.0,
        panel_height,
        COL_MSA_ROW_BG,
        edgecolor='none',
        linewidth=0.0,
    )

    _draw_centered_box_text(
        ax=ax,
        center_x=nt_start + 1.0,
        center_y=aa_center_y,
        box_width=2.52,
        box_height=aa_height,
        facecolor=to_rgba(_aa_color_for_char(aa), alpha=aa_alpha),
        text=aa,
        fontsize=aa_font_size,
        text_color=_aa_color_for_char(aa),
        fontweight='bold',
        linewidth=0.45,
    )

    codon_text = str(codon).upper()
    for nt_offset, ch in enumerate(codon_text[:3]):
        _draw_centered_box_text(
            ax=ax,
            center_x=nt_start + nt_offset,
            center_y=codon_center_y,
            box_width=0.76,
            box_height=codon_height,
            facecolor=_msa_color_for_char(ch),
            text=ch,
            fontsize=codon_font_size,
            text_color=COL_MSA_TEXT,
            fontfamily=NT_FONT_FAMILY,
            linewidth=0.35,
        )


def _get_msa_block_layout(num_records):
    top_gap = 0.02
    tick_box_height = 0.22
    tick_row_height = 0.23
    tick_gap = 0.012
    section_gap = 0.014
    logo_bottom_gap = 0.012
    consensus_panel_height = 0.245
    logo_height = 2.05 / 3.0
    row_panel_height = 0.24
    row_step = row_panel_height
    bottom_gap = 0.01
    aa_tick_top = top_gap
    nt_tick_top = aa_tick_top + tick_row_height + tick_gap
    consensus_top = nt_tick_top + tick_row_height + tick_gap
    logo_y_top = consensus_top + consensus_panel_height + section_gap
    seq_y_start = logo_y_top + logo_height + logo_bottom_gap
    content_bottom = seq_y_start + (num_records * row_step)
    return {
        'top_gap': top_gap,
        'tick_box_height': tick_box_height,
        'tick_row_height': tick_row_height,
        'tick_gap': tick_gap,
        'section_gap': section_gap,
        'logo_bottom_gap': logo_bottom_gap,
        'consensus_panel_height': consensus_panel_height,
        'logo_height': logo_height,
        'row_panel_height': row_panel_height,
        'row_step': row_step,
        'bottom_gap': bottom_gap,
        'aa_tick_top': aa_tick_top,
        'nt_tick_top': nt_tick_top,
        'consensus_top': consensus_top,
        'logo_y_top': logo_y_top,
        'seq_y_start': seq_y_start,
        'content_bottom': content_bottom,
        'total_height': content_bottom + bottom_gap,
    }


def _draw_msa_block(ax, records, msa_summary, start, end, wrap, font_size, y_offset, layout):
    block_len = end - start
    num_records = len(records)
    block_codons = block_len // 3
    start_codon = start // 3
    tick_box_height = layout['tick_box_height']
    consensus_panel_height = layout['consensus_panel_height']
    logo_height = layout['logo_height']
    row_panel_height = layout['row_panel_height']
    row_step = layout['row_step']
    aa_tick_y = y_offset + layout['aa_tick_top'] + (tick_box_height / 2.0)
    nt_tick_y = y_offset + layout['nt_tick_top'] + (tick_box_height / 2.0)
    consensus_top = y_offset + layout['consensus_top']
    consensus_y_center = consensus_top + (consensus_panel_height / 2.0)
    logo_y_top = y_offset + layout['logo_y_top']
    seq_y_start = y_offset + layout['seq_y_start']
    block_bottom = y_offset + layout['content_bottom']

    consensus_aa_font_size = max(font_size - 0.4, 5)
    consensus_codon_font_size = max(font_size * 0.46, 4)
    for codon_offset in range(block_codons):
        nt_start = codon_offset * 3
        abs_codon_idx = start_codon + codon_offset
        _draw_sequence_split_row(
            ax=ax,
            codon=msa_summary['consensus_codons'][abs_codon_idx],
            aa=msa_summary['consensus_aas'][abs_codon_idx],
            nt_start=nt_start,
            row_top=consensus_top,
            panel_height=consensus_panel_height,
            aa_font_size=consensus_aa_font_size,
            codon_font_size=consensus_codon_font_size,
            aa_alpha=0.18,
        )
        _draw_aa_logo(
            ax=ax,
            freqs=msa_summary['aa_logo_frequencies'][abs_codon_idx],
            nt_start=nt_start,
            y_top=logo_y_top,
            logo_height=logo_height,
        )

    aa_font_size = max(font_size - 0.6, 5)
    codon_font_size = max(font_size * 0.42, 4)
    for row_idx, record in enumerate(records):
        row_top = seq_y_start + row_idx * row_step
        for codon_offset in range(block_codons):
            nt_start = codon_offset * 3
            abs_codon_idx = start_codon + codon_offset
            _draw_sequence_split_row(
                ax=ax,
                codon=msa_summary['seq_codons'][row_idx][abs_codon_idx],
                aa=msa_summary['seq_aas'][row_idx][abs_codon_idx],
                nt_start=nt_start,
                row_top=row_top,
                panel_height=row_panel_height,
                aa_font_size=aa_font_size,
                codon_font_size=codon_font_size,
            )

    for rel_idx in range(0, wrap + 1, 3):
        x_pos = rel_idx - 0.5
        ax.plot([x_pos, x_pos], [y_offset + layout['aa_tick_top'], block_bottom], color=COL_MSA_BORDER, linewidth=1.1, zorder=0)

    nt_tick_step = 9 if wrap > 18 else 3
    aa_tick_step = max(1, nt_tick_step // 3)
    for codon_offset in range(0, block_codons, aa_tick_step):
        _draw_boxed_tick(
            ax=ax,
            center_x=codon_offset * 3 + 1,
            center_y=aa_tick_y,
            box_width=2.52,
            box_height=tick_box_height,
            text=str(start_codon + codon_offset + 1),
            fontsize=max(font_size - 1, 5),
        )
    if block_codons > 0 and ((block_codons - 1) % aa_tick_step) != 0:
        _draw_boxed_tick(
            ax=ax,
            center_x=(block_codons - 1) * 3 + 1,
            center_y=aa_tick_y,
            box_width=2.52,
            box_height=tick_box_height,
            text=str(start_codon + block_codons),
            fontsize=max(font_size - 1, 5),
        )

    for rel_idx in range(0, block_len, nt_tick_step):
        _draw_boxed_tick(
            ax=ax,
            center_x=rel_idx,
            center_y=nt_tick_y,
            box_width=0.76,
            box_height=tick_box_height,
            text=str(start + rel_idx + 1),
            fontsize=max(font_size - 1, 5),
        )
    if block_len > 0 and ((block_len - 1) % nt_tick_step) != 0:
        _draw_boxed_tick(
            ax=ax,
            center_x=block_len - 1,
            center_y=nt_tick_y,
            box_width=0.76,
            box_height=tick_box_height,
            text=str(end),
            fontsize=max(font_size - 1, 5),
        )

    y_transform = ax.get_yaxis_transform()
    ax.text(-0.02, aa_tick_y, 'AA site', transform=y_transform, ha='right', va='center', fontsize=max(font_size - 1, 5), clip_on=False, color=COL_MSA_SUBTEXT)
    ax.text(-0.02, nt_tick_y, 'NT site', transform=y_transform, ha='right', va='center', fontsize=max(font_size - 1, 5), clip_on=False, color=COL_MSA_SUBTEXT)
    ax.text(-0.02, consensus_y_center, 'Consensus', transform=y_transform, ha='right', va='center', fontsize=font_size, clip_on=False, color=COL_MSA_TEXT)
    ax.text(-0.02, logo_y_top + (logo_height / 2.0), 'AA frequency', transform=y_transform, ha='right', va='center', fontsize=font_size, clip_on=False, color=COL_MSA_TEXT)
    for row_idx, record in enumerate(records):
        ax.text(
            -0.02,
            seq_y_start + row_idx * row_step + (row_panel_height / 2.0),
            _truncate_text(record.id, 28),
            transform=y_transform,
            ha='right',
            va='center',
            fontsize=font_size,
            clip_on=False,
        )


def _serialize_figure(fig, plotformat):
    if plotformat == 'svg':
        FigureCanvasSVG(fig)
        buffer = io.StringIO()
        fig.savefig(buffer, format='svg', metadata={'Date': None})
        return buffer.getvalue()
    buffer = io.BytesIO()
    fig.savefig(buffer, format=plotformat, metadata={'Date': None})
    return buffer.getvalue()


def _write_serialized_plot(payload, outfile, plotformat):
    if outfile == '-':
        if plotformat == 'svg':
            sys.stdout.write(payload)
            if not payload.endswith('\n'):
                sys.stdout.write('\n')
        else:
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
        return
    if plotformat == 'svg':
        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(payload)
            if not payload.endswith('\n'):
                f.write('\n')
        return
    with open(outfile, 'wb') as f:
        f.write(payload)


def _plot_ambiguous_bar_chart(ax, records, top_n, title):
    if (top_n <= 0) or (len(records) == 0):
        ax.axis('off')
        return
    top_records = sorted(records, key=lambda item: (-item['ambiguous_codons'], item['seq_id']))[:top_n]
    if len(top_records) == 0:
        ax.axis('off')
        return
    counts = np.array([record['ambiguous_codons'] for record in top_records], dtype=float)
    labels = [_truncate_text(record['seq_id'], 22) for record in top_records]
    ypos = np.arange(len(top_records))
    ax.barh(ypos, counts, color=COL_BAR, edgecolor=COL_BAR_EDGE, linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Ambiguous codons')
    ax.grid(axis='x', color='#dfe5ea', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for y, count in zip(ypos, counts):
        ax.text(count + 0.03 * max(1.0, counts.max()), y, str(int(count)), va='center', ha='left', fontsize=9)


def _build_summary_figure(records, site_summaries, kept_sites, ambiguous_by_seq, args):
    width = _int_arg('--width', getattr(args, 'width', DEFAULT_WIDTH), DEFAULT_WIDTH)
    height = _int_arg('--height', getattr(args, 'height', DEFAULT_HEIGHT), DEFAULT_HEIGHT)
    top_n = _nonnegative_int_arg('--top_n', getattr(args, 'top_n', 0), 0)
    title = getattr(args, 'title', '') or 'cdskit plot'
    min_occupancy = validate_fraction(name='--min_occupancy', value=getattr(args, 'min_occupancy', 0.5))
    max_ambiguous_fraction = validate_fraction(
        name='--max_ambiguous_fraction',
        value=getattr(args, 'max_ambiguous_fraction', 1.0),
    )
    drop_stop_codon = bool(getattr(args, 'drop_stop_codon', False))

    num_sequences = len(records)
    num_sites = len(site_summaries)
    removed_sites = num_sites - len(kept_sites)
    summary_text = _build_summary_text(
        num_sequences=num_sequences,
        num_sites=num_sites,
        kept_sites=len(kept_sites),
        removed_sites=removed_sites,
        min_occupancy=min_occupancy,
        max_ambiguous_fraction=max_ambiguous_fraction,
        drop_stop_codon=drop_stop_codon,
    )

    show_bar_chart = (top_n > 0) and (len(ambiguous_by_seq) > 0)
    fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor='white')
    grid_kwargs = {
        'left': 0.08,
        'right': 0.97,
        'top': 0.84,
        'bottom': 0.12,
        'hspace': 0.12,
        'height_ratios': [4.0, 0.35],
    }
    if show_bar_chart:
        gs = fig.add_gridspec(2, 2, width_ratios=[3.8, 1.4], wspace=0.28, **grid_kwargs)
        bar_ax = fig.add_subplot(gs[:, 1])
    else:
        gs = fig.add_gridspec(2, 1, **grid_kwargs)
        bar_ax = None

    main_ax = fig.add_subplot(gs[0, 0])
    strip_ax = fig.add_subplot(gs[1, 0], sharex=main_ax)

    fig.text(0.08, 0.94, title, fontsize=18, fontweight='bold', ha='left', va='top')
    fig.text(0.08, 0.90, summary_text, fontsize=10, ha='left', va='top', color='#607d8b')

    legend_handles = [
        Line2D([0], [0], color=COL_OCCUPANCY, linewidth=2.2, label='occupancy'),
        Line2D([0], [0], color=COL_AMBIGUOUS, linewidth=2.0, label='ambiguity'),
        Line2D([0], [0], color=COL_THRESHOLD, linewidth=1.2, linestyle='--', label='occupancy threshold'),
        Line2D([0], [0], color=COL_AMBIGUOUS, linewidth=1.2, linestyle=':', label='ambiguity threshold'),
        Line2D([0], [0], color=COL_STOP, marker='v', linestyle='None', markersize=6, label='stop codon'),
        Patch(facecolor=COL_KEEP, edgecolor='none', label='kept'),
        Patch(facecolor=COL_REMOVE, edgecolor='none', label='removed'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.97, 0.945), ncol=3, frameon=False, fontsize=9)

    if num_sites == 0:
        main_ax.text(0.5, 0.5, 'No codon sites detected.', ha='center', va='center', transform=main_ax.transAxes)
        main_ax.set_axis_off()
        strip_ax.set_axis_off()
    else:
        x = np.arange(1, num_sites + 1)
        occupancy_values = np.array([site['occupancy'] for site in site_summaries], dtype=float)
        ambiguous_values = np.array([site['ambiguous_fraction'] for site in site_summaries], dtype=float)
        stop_positions = x[[site['stop_codons'] > 0 for site in site_summaries]]
        keep_mask = np.array([[1 if site_idx in set(kept_sites) else 0 for site_idx in range(num_sites)]], dtype=int)

        main_ax.plot(x, occupancy_values, color=COL_OCCUPANCY, linewidth=2.2)
        main_ax.plot(x, ambiguous_values, color=COL_AMBIGUOUS, linewidth=2.0)
        main_ax.axhline(min_occupancy, color=COL_THRESHOLD, linewidth=1.2, linestyle='--')
        main_ax.axhline(max_ambiguous_fraction, color=COL_AMBIGUOUS, linewidth=1.2, linestyle=':')
        if len(stop_positions) > 0:
            main_ax.scatter(stop_positions, np.full(len(stop_positions), 1.04), color=COL_STOP, marker='v', s=32, zorder=4)
        main_ax.set_xlim(0.5, num_sites + 0.5)
        main_ax.set_ylim(-0.02, 1.08)
        main_ax.set_ylabel('Fraction')
        main_ax.set_title('Occupancy and ambiguity by codon site', fontsize=12)
        main_ax.grid(color='#dfe5ea', linewidth=0.8, alpha=0.8)
        main_ax.set_axisbelow(True)
        main_ax.tick_params(axis='x', labelbottom=False)

        strip_ax.imshow(
            keep_mask,
            aspect='auto',
            interpolation='nearest',
            cmap=ListedColormap([COL_REMOVE, COL_KEEP]),
            extent=(0.5, num_sites + 0.5, 0.0, 1.0),
        )
        strip_ax.set_yticks([])
        strip_ax.set_xlabel('Codon site')
        strip_ax.set_title('keep/remove strip', fontsize=10, pad=4)
        strip_ax.set_xlim(0.5, num_sites + 0.5)
        strip_ax.set_xticks(_make_site_ticks(num_sites))
        strip_ax.grid(False)

    if bar_ax is not None:
        _plot_ambiguous_bar_chart(bar_ax, ambiguous_by_seq, top_n, 'Top ambiguous sequences')

    return fig


def _build_map_figure(records, summary, args):
    width = _int_arg('--width', getattr(args, 'width', DEFAULT_WIDTH), DEFAULT_WIDTH)
    base_height = _int_arg('--height', getattr(args, 'height', DEFAULT_HEIGHT), DEFAULT_HEIGHT)
    row_height = _int_arg('--row_height', getattr(args, 'row_height', DEFAULT_ROW_HEIGHT), DEFAULT_ROW_HEIGHT)
    label_width = _int_arg('--label_width', getattr(args, 'label_width', DEFAULT_LABEL_WIDTH), DEFAULT_LABEL_WIDTH)
    top_n = _nonnegative_int_arg('--top_n', getattr(args, 'top_n', 0), 0)
    title = getattr(args, 'title', '') or 'cdskit plot (map)'

    num_records = len(records)
    num_sites = len(summary['site_summaries'])
    height = max(base_height, 180 + max(1, num_records) * row_height)
    left_margin = min(0.42, max(0.12, label_width / max(1.0, float(width))))
    show_bar_chart = (top_n > 0) and (len(summary['sequence_ambiguous_counts']) > 0)

    fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor='white')
    grid_kwargs = {
        'left': left_margin,
        'right': 0.97,
        'top': 0.82,
        'bottom': 0.10,
        'hspace': 0.10,
        'height_ratios': [4.0, 0.35],
    }
    if show_bar_chart:
        gs = fig.add_gridspec(2, 2, width_ratios=[4.0, 1.35], wspace=0.28, **grid_kwargs)
        bar_ax = fig.add_subplot(gs[:, 1])
    else:
        gs = fig.add_gridspec(2, 1, **grid_kwargs)
        bar_ax = None

    map_ax = fig.add_subplot(gs[0, 0])
    strip_ax = fig.add_subplot(gs[1, 0], sharex=map_ax)

    fig.text(0.08, 0.94, title, fontsize=18, fontweight='bold', ha='left', va='top')
    fig.text(
        0.08,
        0.90,
        (
            f'Sequences: {num_records} | Codon sites: {num_sites} | '
            f'Kept sites: {len(summary["kept_sites"])} | Removed sites: {num_sites - len(summary["kept_sites"])}'
        ),
        fontsize=10,
        ha='left',
        va='top',
        color='#607d8b',
    )
    fig.text(
        0.08,
        0.87,
        (
            f'Thresholds: occupancy >= {getattr(args, "min_occupancy", 0.0):.2f}, '
            f'ambiguous fraction <= {getattr(args, "max_ambiguous_fraction", 1.0):.2f}, '
            f'drop stop codon = {bool(getattr(args, "drop_stop_codon", False))}'
        ),
        fontsize=10,
        ha='left',
        va='top',
        color='#607d8b',
    )

    legend_handles = [
        Patch(facecolor=COL_COMPLETE, edgecolor='none', label='complete'),
        Patch(facecolor=COL_MISSING, edgecolor='none', label='missing'),
        Patch(facecolor=COL_AMBIGUOUS, edgecolor='none', label='ambiguous'),
        Patch(facecolor=COL_STOP, edgecolor='none', label='stop'),
        Patch(facecolor=COL_KEEP, edgecolor='none', label='keep'),
        Patch(facecolor=COL_REMOVE, edgecolor='none', label='remove'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.97, 0.945), ncol=3, frameon=False, fontsize=9)

    if (num_records == 0) or (num_sites == 0):
        map_ax.text(0.5, 0.5, 'No sequences to draw', ha='center', va='center', transform=map_ax.transAxes)
        map_ax.set_axis_off()
        strip_ax.set_axis_off()
    else:
        state_matrix = _build_state_matrix(records=records, codontable=args.codontable)
        keep_mask = np.array([[1 if site_idx in set(summary['kept_sites']) else 0 for site_idx in range(num_sites)]], dtype=int)
        site_ticks = _make_site_ticks(num_sites)
        tick_positions = [tick - 1 for tick in site_ticks]
        cmap = ListedColormap([COL_COMPLETE, COL_MISSING, COL_AMBIGUOUS, COL_STOP])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

        map_ax.imshow(state_matrix, aspect='auto', interpolation='nearest', cmap=cmap, norm=norm)
        map_ax.set_xlim(-0.5, num_sites - 0.5)
        map_ax.set_ylabel('Sequence')
        map_ax.set_yticks(np.arange(num_records))
        map_ax.set_yticklabels([_truncate_text(record.id, 28) for record in records])
        map_ax.set_title('Codon-state alignment map', fontsize=12)
        map_ax.tick_params(axis='x', labelbottom=False)

        strip_ax.imshow(
            keep_mask,
            aspect='auto',
            interpolation='nearest',
            cmap=ListedColormap([COL_REMOVE, COL_KEEP]),
            extent=(-0.5, num_sites - 0.5, 0.0, 1.0),
        )
        strip_ax.set_yticks([])
        strip_ax.set_xlabel('Codon site')
        strip_ax.set_title('keep/remove strip', fontsize=10, pad=4)
        strip_ax.set_xlim(-0.5, num_sites - 0.5)
        strip_ax.set_xticks(tick_positions)
        strip_ax.set_xticklabels([str(tick) for tick in site_ticks])

    if bar_ax is not None:
        top_records = [
            {'seq_id': seq_id, 'ambiguous_codons': count}
            for seq_id, count in summary['sequence_ambiguous_counts']
        ]
        _plot_ambiguous_bar_chart(bar_ax, top_records, top_n, 'Top ambiguous codon counts')

    return fig


def _build_msa_figure(records, args):
    width = _int_arg('--width', getattr(args, 'width', DEFAULT_WIDTH), DEFAULT_WIDTH)
    base_height = _int_arg('--height', getattr(args, 'height', DEFAULT_HEIGHT), DEFAULT_HEIGHT)
    row_height = _int_arg('--row_height', getattr(args, 'row_height', DEFAULT_ROW_HEIGHT), DEFAULT_ROW_HEIGHT)
    label_width = _int_arg('--label_width', getattr(args, 'label_width', DEFAULT_LABEL_WIDTH), DEFAULT_LABEL_WIDTH)
    wrap = _int_arg('--wrap', getattr(args, 'wrap', DEFAULT_WRAP), DEFAULT_WRAP)
    title = getattr(args, 'title', '') or 'cdskit plot (msa)'
    if wrap % 3 != 0:
        raise Exception('--wrap must be a multiple of three when --mode=msa. Exiting.\n')

    num_records = len(records)
    num_sites = len(records[0].seq) if len(records) > 0 else 0
    msa_summary = _build_msa_summary(records=records, codontable=args.codontable)
    num_blocks = max(1, (num_sites + wrap - 1) // wrap) if num_sites > 0 else 1
    font_size = max(5, min(11, row_height * 0.48))
    block_layout = _get_msa_block_layout(num_records=num_records)
    block_gap = 0.14
    total_msa_height = (num_blocks * block_layout['total_height']) + (max(0, num_blocks - 1) * block_gap)
    unit_px = max(38.0, (float(row_height) / max(0.01, block_layout['row_panel_height'])) * 0.56)
    height = max(base_height, 130 + int(total_msa_height * unit_px))
    left_margin = min(0.42, max(0.14, label_width / max(1.0, float(width))))
    desired_nt_px = max(6.0, font_size * 0.58)
    grid_width_fraction = min(0.98 - left_margin, (wrap * desired_nt_px) / max(1.0, float(width)))
    right_fraction = min(0.98, left_margin + grid_width_fraction)

    fig = Figure(figsize=(width / 100.0, height / 100.0), dpi=100, facecolor='white')
    msa_ax = fig.add_axes([left_margin, 0.06, right_fraction - left_margin, 0.77])

    fig.text(0.08, 0.94, title, fontsize=18, fontweight='bold', ha='left', va='top')
    fig.text(
        0.08,
        0.90,
        f'{_fmt_count(num_records)} sequences | {_fmt_count(num_sites)} sites | wrap {wrap}',
        fontsize=10,
        ha='left',
        va='top',
        color='#607d8b',
    )
    fig.text(
        0.08,
        0.87,
        'Consensus codon and amino acid plus per-codon amino-acid frequency logo.',
        fontsize=10,
        ha='left',
        va='top',
        color='#607d8b',
    )

    legend_handles = [
        Patch(facecolor=COL_A, edgecolor='none', label='A'),
        Patch(facecolor=COL_C, edgecolor='none', label='C'),
        Patch(facecolor=COL_G, edgecolor='none', label='G'),
        Patch(facecolor=COL_T, edgecolor='none', label='T'),
        Patch(facecolor=COL_N, edgecolor='none', label='N'),
        Patch(facecolor=COL_GAP, edgecolor='#d0d7de', label='gap/missing'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.945), ncol=3, frameon=False, fontsize=9)

    if (num_records == 0) or (num_sites == 0):
        msa_ax.text(0.5, 0.5, 'No sequences to draw', ha='center', va='center', transform=msa_ax.transAxes)
        msa_ax.set_axis_off()
        return fig

    msa_ax.set_xlim(-0.5, wrap - 0.5)
    msa_ax.set_ylim(total_msa_height, 0.0)
    msa_ax.set_xticks([])
    msa_ax.set_yticks([])
    msa_ax.set_frame_on(False)

    for block_idx in range(num_blocks):
        start = block_idx * wrap
        end = min(num_sites, start + wrap)
        _draw_msa_block(
            ax=msa_ax,
            records=records,
            msa_summary=msa_summary,
            start=start,
            end=end,
            wrap=wrap,
            font_size=font_size,
            y_offset=block_idx * (block_layout['total_height'] + block_gap),
            layout=block_layout,
        )

    return fig


def plot_main(args):
    mode = _normalize_mode(getattr(args, 'mode', 'summary'))
    outfile = getattr(args, 'outfile', '-')
    plotformat = _normalize_plotformat(getattr(args, 'plotformat', 'auto'), outfile=outfile)
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    threads = resolve_threads(getattr(args, 'threads', 1))
    stop_if_not_dna(records=records, label='--seqfile')
    stop_if_not_aligned(records=records)
    stop_if_not_multiple_of_three(records=records)
    stop_if_invalid_codontable(args.codontable)
    min_occupancy = validate_fraction(name='--min_occupancy', value=getattr(args, 'min_occupancy', 0.5))
    max_ambiguous_fraction = validate_fraction(
        name='--max_ambiguous_fraction',
        value=getattr(args, 'max_ambiguous_fraction', 1.0),
    )
    _ = _nonnegative_int_arg('--top_n', getattr(args, 'top_n', 0), 0)

    if mode == 'map':
        map_args = copy.copy(args)
        if getattr(map_args, 'title', '') in ('', None):
            map_args.title = 'cdskit plot (map)'
        summary = summarize_draw(
            records=records,
            codontable=args.codontable,
            min_occupancy=min_occupancy,
            max_ambiguous_fraction=max_ambiguous_fraction,
            drop_stop_codon=bool(getattr(args, 'drop_stop_codon', False)),
        )
        fig = _build_map_figure(records=records, summary=summary, args=map_args)
        payload = _serialize_figure(fig=fig, plotformat=plotformat)
        _write_serialized_plot(payload=payload, outfile=outfile, plotformat=plotformat)
        return payload

    if mode == 'msa':
        msa_args = copy.copy(args)
        if getattr(msa_args, 'title', '') in ('', None):
            msa_args.title = 'cdskit plot (msa)'
        fig = _build_msa_figure(records=records, args=msa_args)
        payload = _serialize_figure(fig=fig, plotformat=plotformat)
        _write_serialized_plot(payload=payload, outfile=outfile, plotformat=plotformat)
        return payload

    seq_strings = [str(record.seq) for record in records]
    site_summaries = [
        summarize_codon_site(seq_strings=seq_strings, codon_site=codon_site, codontable=args.codontable)
        for codon_site in range(len(seq_strings[0]) // 3)
    ] if len(records) > 0 else []
    kept_sites = choose_kept_codon_sites(
        site_summaries=site_summaries,
        num_sequences=len(records),
        min_occupancy=min_occupancy,
        max_ambiguous_fraction=max_ambiguous_fraction,
        drop_stop_codon=bool(getattr(args, 'drop_stop_codon', False)),
    )
    ambiguous_by_seq = _compute_sequence_stats(records=records, threads=threads)
    fig = _build_summary_figure(
        records=records,
        site_summaries=site_summaries,
        kept_sites=kept_sites,
        ambiguous_by_seq=ambiguous_by_seq,
        args=args,
    )
    payload = _serialize_figure(fig=fig, plotformat=plotformat)
    _write_serialized_plot(payload=payload, outfile=outfile, plotformat=plotformat)
    return payload
