#!/bin/bash
set -euo pipefail
echo "=== pattern_matcher.py lines 305-325 ==="
sed -n '305,325p' /usr/local/lib/python3.12/dist-packages/torch/_inductor/pattern_matcher.py 2>/dev/null || echo "NOT FOUND"
echo "=== post_grad.py lines 345-375 ==="
sed -n '345,375p' /usr/local/lib/python3.12/dist-packages/torch/_inductor/fx_passes/post_grad.py 2>/dev/null || echo "NOT FOUND"
echo "=== post_grad.py lines 1240-1260 ==="
sed -n '1240,1260p' /usr/local/lib/python3.12/dist-packages/torch/_inductor/fx_passes/post_grad.py 2>/dev/null || echo "NOT FOUND"
echo "=== DONE ==="
