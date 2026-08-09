import argparse
import math
import os
import sys


TRUE_VALUES = frozenset(['y', 'yes', 't', 'true', 'on', '1'])
FALSE_VALUES = frozenset(['n', 'no', 'f', 'false', 'off', '0'])

_TOKEN_LABELS = {
    'ctp': 'cTP',
    'deeploc': 'DeepLoc',
    'dl': 'deep-learning',
    'esm': 'ESM',
    'fasta': 'FASTA',
    'json': 'JSON',
    'ltp': 'lTP',
    'md': 'Markdown',
    'mtp': 'mTP',
    'notp': 'noTP',
    'npz': 'NPZ',
    'oof': 'out-of-fold',
    'rnn': 'RNN',
    'sp': 'SP',
    'tsv': 'TSV',
}

_COMMON_ARGUMENT_HELP = {
    'device': 'Computation device, such as cpu, cuda, or auto.',
    'random_state': 'Random seed passed to the estimator.',
    'seed': 'Random seed for reproducible processing.',
    'threads': 'Number of worker threads; 0 auto-detects CPUs up to the safety limit.',
    'verbose': 'Enable detailed progress output.',
}

_FILE_SUFFIXES = (
    ('_predictions_tsv', 'predictions TSV file'),
    ('_tsvs', 'TSV files'),
    ('_fasta', 'FASTA file'),
    ('_tsv', 'TSV file'),
    ('_npzs', 'NPZ archives'),
    ('_npz', 'NPZ archive'),
    ('_json', 'JSON file'),
    ('_md', 'Markdown file'),
    ('_tab', 'tabular file'),
    ('_dir', 'directory'),
    ('_path', 'file'),
    ('_prefix', 'file prefix'),
)

_PARAMETER_HELP_RULES = (
    (('min_seq_id',), 'Minimum sequence-identity fraction used for {subject}.'),
    (('coverage',), 'Minimum alignment-coverage fraction used for {subject}.'),
    (('cov_mode',), 'Coverage calculation mode used for {subject}.'),
    (('threads',), 'Number of worker threads used for {subject}.'),
    (('model_name',), 'Pretrained model name or identifier used for {subject}.'),
    (('seq_len', 'max_len'), 'Maximum sequence length used for {subject}.'),
    (('learning_rate', 'lr'), 'Optimizer learning rate used for {subject}.'),
    (
        ('weight_decay', 'l2', 'l2_regularization'),
        'Regularization strength used for {subject}.',
    ),
    (
        ('max_iter', 'epochs', 'patience_epochs'),
        'Number of training or optimization iterations for {subject}.',
    ),
    (('batch_size',), 'Number of examples in each {subject} batch.'),
    (
        ('hidden_dim', 'embed_dim', 'hidden_rnn', 'hidden_fc'),
        'Hidden representation size used for {subject}.',
    ),
    (
        ('n_filters', 'n_attention', 'attention_size'),
        'Model capacity value used for {subject}.',
    ),
    (('grad_clip_norm',), 'Gradient clipping norm used for {subject}.'),
    (('keep_prob',), 'Retention probability used for {subject}.'),
    (('seed_offset',), 'Offset added to random seeds for {subject}.'),
    (('temperature',), 'Temperature used for {subject}.'),
    (('power',), 'Exponent used for {subject}.'),
    (('alpha',), 'Interpolation weight used for {subject}.'),
    (('base_index',), 'Zero-based model index used for {subject}.'),
    (('timeout_sec',), 'Timeout in seconds for {subject}.'),
)

_SUFFIX_HELP_RULES = (
    (('_max_frac', '_max_count'), 'Maximum limit used for {subject}.'),
    (('_label',), 'Label stored for {subject}.'),
    (
        ('_seed', '_random_state', '_random_states'),
        'Random seed value or values used for {subject}.',
    ),
    (('_n_estimators', 'n_estimators'), 'Number of estimators used for {subject}.'),
    (
        ('_score_min', '_score_max', '_score_step', '_score_steps'),
        'Score-search range value used for {subject}.',
    ),
    (('_grid_step',), 'Step size used for the {subject} search grid.'),
    (
        ('_min_samples_leaf',),
        'Minimum samples per estimator leaf for {subject}.',
    ),
    (('_max_features',), 'Maximum feature subset used for {subject}.'),
    (
        ('_classes', '_weights', '_grid', '_sizes', '_folds'),
        'Values used for {subject}.',
    ),
)

_FALLBACK_SUFFIX_HELP_RULES = (
    (
        ('_epochs', '_batch_size', '_num_layers', '_num_filters'),
        'Training value for {subject}.',
    ),
    (
        ('_rate', '_weight', '_fraction', '_dropout', '_alpha'),
        'Numeric value used for {subject}.',
    ),
)


def _humanize_option_name(name):
    return ' '.join(_TOKEN_LABELS.get(token, token) for token in name.split('_'))


def _matches_parameter(name, *parameters):
    return any(name == value or name.endswith('_' + value) for value in parameters)


def _file_argument_help(name):
    for suffix, file_kind in _FILE_SUFFIXES:
        if not name.endswith(suffix):
            continue
        stem = name[:-len(suffix)].strip('_')
        output = (
            stem == 'out'
            or stem.startswith('out_')
            or stem.endswith('_out')
            or '_out_' in stem
            or 'report' in stem
            or 'comparison' in stem
            or name.endswith('_prefix')
        )
        if stem == 'out':
            stem = ''
        else:
            stem = stem.removeprefix('out_').removesuffix('_out').strip('_')
        purpose = _humanize_option_name(stem) if stem else 'result'
        if file_kind == 'directory':
            if output:
                return 'Directory where {} outputs are written.'.format(purpose)
            return 'Directory containing the {} inputs.'.format(purpose)
        if output:
            return 'Path for the output {} {}.'.format(purpose, file_kind)
        return 'Path to the {} {}.'.format(purpose, file_kind)
    return None


def _rule_argument_help(name, subject):
    for parameters, template in _PARAMETER_HELP_RULES:
        if _matches_parameter(name, *parameters):
            return template.format(subject=subject)
    for suffixes, template in _SUFFIX_HELP_RULES:
        if name.endswith(suffixes):
            return template.format(subject=subject)
    return None


def _fallback_suffix_argument_help(name, subject):
    for suffixes, template in _FALLBACK_SUFFIX_HELP_RULES:
        if name.endswith(suffixes):
            return template.format(subject=subject)
    return None


def _automatic_argument_help(option, kwargs):
    """Build useful fallback help for research CLIs with many parameters."""
    name = option.lstrip('-').replace('-', '_')
    if kwargs.get('action') == 'version':
        return 'Show the program version and exit.'
    if name in _COMMON_ARGUMENT_HELP:
        return _COMMON_ARGUMENT_HELP[name]

    subject = _humanize_option_name(name)
    action = kwargs.get('action')
    if action in {'store_true', 'store_false'}:
        verb = 'Enable' if action == 'store_true' else 'Disable'
        return '{} {}.'.format(verb, subject)
    if kwargs.get('type') is parse_bool:
        return 'Enable or disable {}.'.format(subject)

    file_help = _file_argument_help(name)
    if file_help is not None:
        return file_help

    if name == 'mmseqs' or name.endswith('_mmseqs'):
        return 'Path or command name for the {} executable.'.format(subject)
    if (
        name in {'model', 'model_a', 'model_b', 'resume_state'}
        or name.endswith('_model')
    ):
        return 'Path to the {} file.'.format(subject)
    if name.endswith('_out'):
        return 'Path for the output {} file.'.format(
            _humanize_option_name(name[:-4]) or 'result'
        )
    if kwargs.get('choices') is not None:
        return 'Method or mode used for {}.'.format(subject)
    if 'threshold' in name:
        return 'Decision threshold or candidate thresholds for {}.'.format(subject)
    rule_help = _rule_argument_help(name, subject)
    if rule_help is not None:
        return rule_help
    if name.startswith('num_'):
        return 'Number of {}.'.format(_humanize_option_name(name[4:]))
    if name.startswith('max_'):
        return 'Maximum value allowed for {}.'.format(_humanize_option_name(name[4:]))
    if name.startswith('min_'):
        return 'Minimum value allowed for {}.'.format(_humanize_option_name(name[4:]))
    fallback_help = _fallback_suffix_argument_help(name, subject)
    if fallback_help is not None:
        return fallback_help
    return 'Value used for {}.'.format(subject)


def parse_bool(value):
    """Parse the boolean spelling accepted consistently by every cdskit CLI."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError(
        'expected one of yes/no, true/false, on/off, or 1/0'
    )


def finite_float(value):
    """Parse a finite floating-point CLI value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('expected a floating-point number')
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError('expected a finite floating-point number')
    return parsed


def warn_deprecated_option(old_option, new_option, stream=None):
    if stream is None:
        stream = sys.stderr
    stream.write(
        'Warning: {} is deprecated; use {} instead.\n'.format(
            old_option,
            new_option,
        )
    )


def resolve_threads(threads):
    """Resolve threads; 0 auto-detects CPUs without exceeding the safety limit."""
    if threads is None:
        return 1
    value = int(threads)
    if value < 0:
        raise ValueError('--threads should be >= 0. 0 means auto-detect CPU count.')
    maximum = int(os.environ.get('CDSKIT_MAX_THREADS', '64'))
    if maximum < 1:
        raise ValueError('CDSKIT_MAX_THREADS should be >= 1.')
    if value == 0:
        return min(maximum, max(1, int(os.cpu_count() or 1)))
    if value > maximum:
        raise ValueError(
            '--threads should be <= {}. Set CDSKIT_MAX_THREADS to raise the safety limit.'.format(
                maximum
            )
        )
    return value


class DeprecatedStoreTrueAction(argparse.Action):
    """Compatibility action for a deprecated flag that stores True."""

    def __init__(self, option_strings, dest, replacement, **kwargs):
        self.replacement = replacement
        kwargs.setdefault('nargs', 0)
        kwargs.setdefault('default', False)
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        warn_deprecated_option(option_string, self.replacement)
        setattr(namespace, self.dest, True)


class DeprecatedNegatedBooleanAction(argparse.Action):
    """Compatibility action mapping ``--no_foo BOOL`` to ``--foo !BOOL``."""

    def __init__(self, option_strings, dest, replacement, **kwargs):
        self.replacement = replacement
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        warn_deprecated_option(option_string, self.replacement)
        setattr(namespace, self.dest, not bool(values))


class CdskitArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with shared defaults and deprecated-option aliases."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('formatter_class', argparse.ArgumentDefaultsHelpFormatter)
        super().__init__(*args, **kwargs)
        self._deprecated_aliases = {}

    def add_deprecated_alias(self, old_option, new_option):
        if not old_option.startswith('--') or not new_option.startswith('--'):
            raise ValueError('Deprecated aliases should be long options.')
        self._deprecated_aliases[old_option] = new_option

    def add_argument(self, *args, **kwargs):
        if kwargs.get('type') is float:
            kwargs['type'] = finite_float
        if (
            'help' not in kwargs
            and any(isinstance(value, str) and value.startswith('-') for value in args)
        ):
            long_options = [
                value for value in args
                if isinstance(value, str) and value.startswith('--')
            ]
            label = long_options[0] if long_options else str(args[0])
            kwargs['help'] = _automatic_argument_help(label, kwargs)
        return super().add_argument(*args, **kwargs)

    def _normalize_deprecated_aliases(self, args):
        if args is None:
            normalized = list(sys.argv[1:])
        else:
            normalized = list(args)
        warned = set()
        for index, token in enumerate(normalized):
            option, separator, value = str(token).partition('=')
            replacement = self._deprecated_aliases.get(option)
            if replacement is None:
                continue
            if option not in warned:
                warn_deprecated_option(option, replacement)
                warned.add(option)
            normalized[index] = replacement + (separator + value if separator else '')
        return normalized

    def parse_known_args(self, args=None, namespace=None):
        normalized = self._normalize_deprecated_aliases(args)
        return super().parse_known_args(normalized, namespace)


def add_deprecated_aliases(parser, aliases):
    for old_option, new_option in aliases.items():
        parser.add_deprecated_alias(old_option, new_option)
