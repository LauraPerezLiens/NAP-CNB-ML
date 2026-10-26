#!/usr/bin/env python3
"""Plot selected per-chunk metrics without importing the training code or PyTorch."""
import argparse
import json
import math
from collections import deque
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_RUN = BASE / 'chunk_ladder100_layernorm_cosine_shuffled_allscaler_dualeval_100epochs_1GPU' / 'chunks_100'
ALIASES = {'acc': 'accuracy', 'auc': 'roc_auc', 'bal_acc': 'balanced_accuracy'}
SERIES = {'train_online': 'TRAIN ONLINE', 'train_eval': 'TRAIN EVAL', 'test_eval': 'TEST EVAL',
          'train_eval_stage': 'TRAIN EVAL STAGE', 'test_eval_stage': 'TEST EVAL STAGE',
          'train_eval_all': 'TRAIN EVAL ALL', 'test_eval_all': 'TEST EVAL ALL'}


def load_history(path, series, metrics, epochs=None):
    """Keep selected numeric values only; the latest replay of a chunk wins."""
    records = {}
    incomplete = 0
    with path.open() as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                if not line.endswith('\n'):
                    incomplete += 1
                    continue
                raise ValueError(f'{path}:{number}: invalid JSON') from error
            epoch, chunk = int(row['epoch']), int(row['chunk_offset'])
            if epochs and epoch not in epochs:
                continue
            values = {}
            for name in series:
                if name not in row:
                    raise ValueError(f'Series {name!r} missing at line {number}')
                for metric in metrics:
                    if metric not in row[name]:
                        available = ', '.join(row[name])
                        raise ValueError(f'Unknown metric {metric!r}. Available: {available}')
                    values[name, metric] = float(row[name][metric])
            records[epoch, chunk] = values
    if not records:
        raise ValueError('No matching per-chunk records found.')
    if incomplete:
        print('Ignored an incomplete last record; training may be writing it.')
    return sorted(records.items())


def rolling_mean(values, window):
    """Trailing, unweighted mean of finite per-chunk measurements."""
    queue = deque()
    total = 0.0
    count = 0
    result = []
    for value in values:
        queue.append(value)
        if math.isfinite(value):
            total += value
            count += 1
        if len(queue) > window:
            previous = queue.popleft()
            if math.isfinite(previous):
                total -= previous
                count -= 1
        result.append(total / count if count else float('nan'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_RUN,
                        help='Run directory or monitor_history.jsonl file.')
    parser.add_argument('--metrics', nargs='+', default=['loss'],
                        help='E.g. loss accuracy recall specificity roc_auc; acc/auc aliases accepted.')
    parser.add_argument('--series', nargs='+', choices=SERIES, default=['train_online'])
    parser.add_argument('--separate-series', action='store_true',
                        help='One panel per series and metric, with shared training x axis.')
    parser.add_argument('--epochs', nargs='+', type=int, help='Only plot these epochs.')
    parser.add_argument('--window', type=int, default=1, help='Trailing mean in chunks; default 1 plots every raw value without averaging.')
    parser.add_argument('--chunks-per-epoch', type=int, default=100,
                        help='Used to build a continuous x axis across epochs.')
    parser.add_argument('--output', type=Path, help='Output PNG, SVG or PDF path.')
    parser.add_argument('--no-raw', action='store_true', help='Show only the smoothed curves.')
    args = parser.parse_args()
    if args.window < 1 or args.chunks_per_epoch < 1:
        parser.error('--window and --chunks-per-epoch must be positive.')
    metrics = list(dict.fromkeys(ALIASES.get(m, m) for m in args.metrics))
    series = list(dict.fromkeys(args.series))
    path = args.input / 'monitor_history.jsonl' if args.input.is_dir() else args.input
    try:
        records = load_history(path, series, metrics, args.epochs)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if any(chunk > args.chunks_per_epoch for (epoch, chunk), _ in records):
        parser.error('A chunk offset exceeds --chunks-per-epoch.')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    suffix = '_panels' if args.separate_series else ''
    output = args.output or path.parent / 'plots' / f'{"_".join(series)}_{"_".join(metrics)}{suffix}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = [(metric, [name]) for metric in metrics for name in series] if args.separate_series else [(metric, series) for metric in metrics]
    figure, axes = plt.subplots(len(panels), 1, figsize=(13, 3.5 * len(panels)),
                                sharex=True, squeeze=False, layout='constrained')
    epochs = sorted({key[0] for key, values in records})
    colors = {'train_online': '#2563eb', 'train_eval': '#059669', 'test_eval': '#e76f21',
              'train_eval_stage': '#059669', 'test_eval_stage': '#e76f21',
              'train_eval_all': '#8b5cf6', 'test_eval_all': '#dc2626'}
    for (metric, panel_series), axis in zip(panels, axes[:, 0]):
        for name in panel_series:
            first = True
            for epoch in epochs:
                selected = [(key, values) for key, values in records if key[0] == epoch]
                x = [(epoch - 1) * args.chunks_per_epoch + key[1] for key, _ in selected]
                y = [values[name, metric] for _, values in selected]
                if not args.no_raw and args.window > 1:
                    axis.plot(x, y, color=colors[name], alpha=.18, linewidth=.6,
                              label=f'{SERIES[name]} — per chunk' if first else None,
                              rasterized=True)
                axis.plot(x, rolling_mean(y, args.window), color=colors[name], linewidth=1.5,
                          label=f'{SERIES[name]} — mean {args.window} chunks' if args.window > 1 and first
                          else SERIES[name] if first else None)
                first = False
        for epoch in epochs[1:]:
            boundary = (epoch - 1) * args.chunks_per_epoch + .5
            axis.axvline(boundary, color='#777777', linestyle='--', linewidth=.8)
            axis.text(boundary, .98, f' Epoch {epoch}', transform=axis.get_xaxis_transform(),
                      va='top', fontsize=9)
        axis.set_ylabel(metric)
        axis.grid(alpha=.2)
        axis.legend(loc='best', fontsize=9)
    # Panels for the same metric use a common y range; different metrics retain their units.
    if args.separate_series:
        for metric in metrics:
            matching = [axis for (panel_metric, _), axis in zip(panels, axes[:, 0])
                        if panel_metric == metric]
            lower = min(axis.get_ylim()[0] for axis in matching)
            upper = max(axis.get_ylim()[1] for axis in matching)
            for axis in matching:
                axis.set_ylim(lower, upper)
    axes[0, 0].set_title(f'Training metrics — {path.parent.name}\n'
                         f'{len(records):,} chunks recorded; last: epoch {records[-1][0][0]}, '
                         f'chunk {records[-1][0][1]:,}')
    axes[-1, 0].set_xlabel('Cumulative training chunks')
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f'Saved: {output.resolve()}')
    if args.window > 1:
        print('Smoothing restarts at each epoch; it is an unweighted mean of per-chunk metrics.')
    else:
        print('All recorded per-chunk values are plotted without averaging.')
    print('TRAIN ONLINE describes the current chunk during training, not a fixed evaluation population.')


if __name__ == '__main__':
    main()
