# NEL jobs (`type: nel`)

Use this only when the task names NEL.

- **Check:** `nel status <id>`.

```bash
extract_nel_state() {
  local jid="$1" nel_bin="${NEL:-nel}" output state_col
  output=$("$nel_bin" status "$jid" 2>&1)
  state_col=$(echo "$output" \
    | awk -F'|' -v prefix="$jid." 'index($1, prefix) == 1 { print $2; exit }')
  [ -z "$state_col" ] && state_col="$output"
  echo "$state_col" \
    | LC_ALL=C tr '[:lower:]' '[:upper:]' \
    | awk 'match($0, /(PENDING|RUNNING|SUCCESS|FAILED|KILLED|ERROR|NOT[[:space:]]+FOUND)/) { print substr($0, RSTART, RLENGTH); exit }' \
    | sed 's/[[:space:]][[:space:]]*/ /g'
}

is_nel_terminal() {
  case "$(extract_nel_state "$1")" in
    SUCCESS|FAILED|KILLED|ERROR|"NOT FOUND") return 0 ;;
    *) return 1 ;;
  esac
}
```

- **On completion:** `nel info <id>` to fetch results.
- **On failure:** `nel info <id> --logs` then inspect server/client/SLURM logs via SSH.

- **Recent runs when no ID is given:** `nel ls runs --since 1d`.
