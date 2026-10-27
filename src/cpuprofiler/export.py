"""Write the best configuration as an executable shell script."""
from dataclasses import replace
from pathlib import Path
import shlex

from .benchmark import BenchmarkResult, execution


def export_tuned(result: BenchmarkResult, output: Path) -> Path:
    output = output.resolve()
    config = replace(result.config, **{
        name: getattr(result.config, name).resolve()
        for name in ('binary', 'model', 'tokenizer')
    })
    if output in (config.binary, config.model, config.tokenizer):
        raise ValueError('output must not overwrite the binary, model or tokenizer')
    command, env = execution(config)
    lines = [
        '#!/bin/bash',
        'set -eu',
        '',
        '# Clear inherited settings, just as in the benchmark.',
        'for name in ${!OMP_@} ${!GOMP_@} ${!OUTPUT_@}; do',
        '    unset "$name"',
        'done',
        '',
    ]
    lines += [f'export {name}={shlex.quote(value)}' for name, value in env.items()
              if name.startswith(('OMP_', 'GOMP_'))]
    lines += ['', 'exec ' + shlex.join(command) + ' "$@"', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines))
    output.chmod(0o755)
    return output
