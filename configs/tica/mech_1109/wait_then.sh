#!/bin/bash
# wait_then.sh FILE PATTERN CMD...: wait until FILE exists or no process matching PATTERN runs on this node, then run CMD.
# Write PATTERN so it does not match its own text in this argv, e.g. "tica_switch_eval[.]py".
F=$1; P=$2; shift 2
sleep 120
until [ -s "$F" ] || ! pgrep -u "$USER" -f "$P" > /dev/null; do sleep 60; done
exec "$@"
