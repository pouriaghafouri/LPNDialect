#!/usr/bin/env python3

"""Minimal harness to run lpn-opt and pipe output through FileCheck."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tool", required=True, help="Path to lpn-opt executable")
  parser.add_argument("--filecheck", required=True, help="Path to FileCheck binary")
  parser.add_argument("--input", required=True, help="MLIR test file")
  parser.add_argument(
      "--tool-arg",
      action="append",
      default=[],
      dest="tool_args",
      help="Additional argument forwarded to lpn-opt (may be repeated)",
  )
  args = parser.parse_args()

  tool_cmd = [args.tool, *args.tool_args, args.input]
  tool = subprocess.run(
      tool_cmd,
      capture_output=True,
      text=True,
      check=False,
  )
  if tool.stderr:
    sys.stderr.write(tool.stderr)

  if tool.returncode != 0:
    sys.stderr.write("\n=== lpn-opt command failed ===\n")
    sys.stderr.write(tool.stdout)
    return tool.returncode

  check = subprocess.run(
      [args.filecheck, args.input],
      input=tool.stdout,
      capture_output=True,
      text=True,
      check=False,
  )
  if check.returncode != 0:
    sys.stderr.write(check.stdout)
    sys.stderr.write(check.stderr)
    sys.stderr.write("\n=== IR fed to FileCheck ===\n")
    sys.stderr.write(tool.stdout)
    return check.returncode

  return 0


if __name__ == "__main__":
  sys.exit(run())
