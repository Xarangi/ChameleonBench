#!/bin/bash
# Two-way Telegram for a Claude Code session, straight on the Bot API with curl (no channels feature).
#   tg.sh send TEXT...   send TEXT (or stdin if no TEXT) to the owner; prints the HTTP status (200 = sent)
#   tg.sh get            print the owner's pending messages, one per line, and mark them read
#   tg.sh once           block until the owner writes, print it, exit (run in the background: wakes the
#                        session only when a message arrives; restart it after each one)
#   tg.sh wait           long-poll forever (50 s requests); print each owner message as one line the
#                        instant it arrives -- run it under the Monitor tool, one event per message
# Setup (once per machine; the token never goes in git):
#   mkdir -p ~/.config/q9b && cp scripts/telegram/tg.sh ~/.config/q9b/ && chmod 700 ~/.config/q9b/tg.sh
#   printf 'TG_TOKEN=%s\n' '<bot token>' > ~/.config/q9b/tg.env && chmod 600 ~/.config/q9b/tg.env
#   optional: printf 'TG_OWNER=%s\n' '<your chat id>' >> ~/.config/q9b/tg.env  (else the first chat to
#   message the bot is pinned as the owner: send it /start yourself right after setup)
# Only the owner's messages are printed; every other chat is ignored. Only one getUpdates consumer may
# poll a bot at a time, so stop any other poller of the same bot (e.g. the telegram plugin) first.
set -uo pipefail
DIR=${TG_DIR:-$HOME/.config/q9b}
# shellcheck disable=SC1091
source "$DIR/tg.env"
API="https://api.telegram.org/bot$TG_TOKEN"
OWNER_F=$DIR/tg.owner
OFF_F=$DIR/tg.offset
[ -n "${TG_OWNER:-}" ] && [ ! -s "$OWNER_F" ] && echo "$TG_OWNER" > "$OWNER_F"
PY=${TG_PY:-/usr/bin/python3}

# One getUpdates call; prints owner messages (newlines -> " / "), pins the first chat as owner,
# advances the stored offset. $1 = long-poll timeout in seconds.
poll() {
  local off resp
  off=$(cat "$OFF_F" 2>/dev/null || echo 0)
  resp=$(curl -sS --max-time $(( $1 + 15 )) "$API/getUpdates" \
    --data-urlencode "offset=$off" --data-urlencode "timeout=$1" \
    --data-urlencode 'allowed_updates=["message"]') || return 1
  "$PY" - "$resp" "$OWNER_F" "$OFF_F" <<'EOF'
import json, sys
from pathlib import Path
resp, owner_f, off_f = json.loads(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
if not resp.get("ok"):
    print(f"tg.sh: getUpdates error {resp.get('error_code')}: {resp.get('description')}", file=sys.stderr)
    sys.exit(1)
owner = owner_f.read_text().strip() if owner_f.exists() else ""
last = None
for u in resp.get("result", []):
    last = u["update_id"]
    m = u.get("message") or {}
    chat = str((m.get("chat") or {}).get("id", ""))
    if not chat:
        continue
    if not owner:  # first chat to write is the owner
        owner = chat
        owner_f.write_text(owner + "\n")
    if chat != owner:
        continue
    text = m.get("text") or m.get("caption") or ("[non-text message]" if m else "")
    if text:
        print(" / ".join(text.splitlines()), flush=True)
if last is not None:
    off_f.write_text(str(last + 1) + "\n")
EOF
}

case "${1:-}" in
  send)
    shift
    owner=$(cat "$OWNER_F" 2>/dev/null)
    [ -n "$owner" ] || { echo "tg.sh: no owner yet (send the bot /start, then run tg.sh get)" >&2; exit 1; }
    if [ $# -gt 0 ]; then text="$*"; else text=$(cat); fi
    code=000
    # Telegram caps a message at 4096 characters: send in chunks
    while [ -n "$text" ]; do
      chunk=${text:0:4000}; text=${text:4000}
      code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 "$API/sendMessage" \
        --data-urlencode "chat_id=$owner" --data-urlencode "text=$chunk")
    done
    echo "$code"
    ;;
  get)
    poll 0
    ;;
  wait)
    while true; do
      poll 50 || sleep 5
    done
    ;;
  once)  # block until the owner writes, print the message(s), exit: a background job that wakes the session
    while true; do
      out=$(poll 50) || { sleep 5; continue; }
      if [ -n "$out" ]; then echo "$out"; exit 0; fi
    done
    ;;
  *)
    echo "usage: tg.sh send TEXT... | get | wait | once" >&2
    exit 2
    ;;
esac
