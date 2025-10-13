"""xform-auditor CLI エントリポイント。"""

from __future__ import annotations

import sys
from typing import Sequence

import click

from .auditor import audit
from .discover import DiscoveryError
from .report import render_json, render_text


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("targets", nargs=-1)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="出力フォーマットを指定する",
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="互換性のためのダミーオプション（現状カラー出力は未実装）",
)
@click.pass_context
def cli(
    ctx: click.Context, targets: tuple[str, ...], output_format: str, no_color: bool
) -> None:
    """指定したモジュールまたはディレクトリ内の Transform を監査する。"""

    _ = no_color  # 互換性用オプション、現状は未使用

    if not targets:
        raise click.UsageError("At least one module or path must be provided.")

    try:
        report = audit(targets)
    except DiscoveryError as exc:
        raise click.ClickException(str(exc)) from exc

    output = (
        render_json(report) if output_format.lower() == "json" else render_text(report)
    )
    click.echo(output)
    ctx.exit(report.summary.exit_code)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else None
    try:
        result = cli.main(args=args, standalone_mode=False)
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return exc.exit_code
    except SystemExit as exc:  # pragma: no cover - click 内部からの SystemExit
        return int(exc.code or 0)
    if result is None:
        return 0
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
