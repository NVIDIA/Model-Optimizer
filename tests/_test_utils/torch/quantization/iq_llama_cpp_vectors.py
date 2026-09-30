# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conformance vectors captured from a real llama.cpp-quantized checkpoint.

Each entry holds packed block bytes lifted verbatim from
unsloth/Qwen3.8-27B-GGUF (Qwen3.8-27B-UD-IQ1_S.gguf) together with the values
llama.cpp's own dequantize_row_* produces for them, computed from
ggml-quants.c revision 9b05354ec6fb58b4e665e9a39ebc40285c015638.

These pin our decoders against bytes we did not produce. A decoder that drifts
from the GGML layout -- a mis-set high bit, a swapped scale nibble, a sign
parity mistake -- fails here even though a round-trip test against our own
encoder would still pass.
"""

import base64
import zlib

import numpy as np

_VECTORS = {
    "iq1_s": {
        "source": "blk.0.attn_qkv.weight",
        "block_bytes": 50,
        "blocks": (
            "eNoBLAHT/sgYPIuVHR5unIa25z3nTMBrrDsowEZpEmhZYzdCh6ybWSOReiY2MD9xWx/BZF4J3Z/LTxnt0w2T+DYRwT0N"
            "IIwJoKkC4mejy6P3sqbJWgv0FgfWRjO1zs0h+ZpZymO6MyQ9Ms9aF+DLPrjwY54mJRLDZo2emJfnx3B7rQMfrk8bWdoJ"
            "Hc7n5ue6duVmzL9SXOzWw0SmcOsXFEu0ESDUFxuXCM0YGhrzGpYf6sJ0hTc5bGk4hFUG9NtWPKTaovj92oteeEPl9Fc9"
            "8BYIR2j/8Hv38eeyZ9fzxt60f8/DEfda/DyI7XWEZ6CXKNxq2nbhUvHVwm4FckftMeVoGA2YAjcSngU6wBLrdmQNavPg"
            "js7fDIkqa+Domf9cZoy23tsEc05IMcBUzWTS8th0Vso1lo8="
        ),
        "expected": (
            "eNqFVzGIHVUUHTCRgGCaLQJugoFUQYLoFuK8T3YtxDKIBASRDRZiRLCwsRtCYrGsWBjEwuAvAiJJJTEE5wW+pAsJNmJC"
            "QNlOkG3EJiiic+7OmT3v/jv/F4d77znn3vf+vPdgtzrwT64+/KwO4TXWiKpFPvVGnrePTyy+dj5XX32RDMhZL+N3DtYF"
            "6EGuPeQYfc41xtb0Pl1zzBftTXldc+VWNmy+XlfXd5NFQjXWqsPPHkSds2gWc3iefa4dcPXH/d+PWnPW9CBqr86gx/cQ"
            "9AD+d3Nf0d49H30zP4t9GpnveVvD9d1sceVWyZFnVIBTv+/3oKZr+W+v39p/Z+qei7DsnIZZK23z0jdZowJcc/hC8rxp"
            "wo96+n6dbdz+mqlaPd0SWjNv/vslA+rzfOFl7me7Gaa/8nU7hvWTL95mPnvzqckiL/Qxj+e1Xn/niUl1+b2asTl3L7Nm"
            "ztp7fO7ryD+HJ7dCNFt/5oifvfrMxHsiL3zqJQcvwLy7AzVh54F4+ALuTNbc0OlzUJ/Uxbye1xnao371RXvyPdEcj2Kf"
            "fr3+vkZ3Uzn1Rve88DpfMYO/n/WhB7k60ekdmtMd3+daw0MMNTXxsFYtrHXNv2e52n6rBpqjj9OQ93yhCwcv/Vb3nNZz"
            "fehhL7lDD7C32kP5bs/ZMOKln76C7/vIz3mOriXDw/s/DIjqz9/NA88Ijh7q5HSu5k5vLn6aq3y8bv76PiES5KmhZlTd"
            "uB5+TjEjyPt56M2IHdda3eesmVvdeSNedc4Mda/he7xxoGZsfrucgIEXrYiEr5XX/ojDeu6eF3eYuYsew/0feyvCz72t"
            "G7++3Py7sRfPf5wAcqwLKN/3RX7lOLuI1F441QKz979MAGsFtfXHdzJqRPWS9z2LfAO/9cEcplcO3gZYzx7+lCKf6gT7"
            "tJ95OEf2HO5Pfgt1avwurKPvsGx+xyHWjECnZc1Zw+M18tqvmu/X9Wze2WmqPnm6RmQOdHcD9yOzVg/4KNd+nct5Omvg"
            "5Pz0nPSsNIfHa9pLzp+5+vR+8T0U70jekmpzb1XeV+Qr3lrgtXpzra527+7FZaAP8ecr2XJGArVyWus64FCv3uz+PzlT"
            "G078kYacvOoRxx5E5RfNVNy4lgzbl7KBuUafA9Mj9QBomvs68nDm9Ejb5dmwfSkZmCNCVx41AY45Z6hfdeXU68+G5+XP"
            "NTpP9fBORPfBz9d7FJ0TzyriNKfH3we9J9GZ64zdu9hzKqDc5lo7RAI6o9foJx95yY316jrarzW9vh6by171oN7ZaAtc"
            "/TZZXH2ULQeQs6bm+9hLD2ud4zXE5y92fw/9ng03VyZWK8D7HD72eH+EMR/4e1vdOzhWGzQHPvouGVTzfuiLIn0E5+6v"
            "AT5ZnB5rLUcEyCmoax+i9lFnv/d43q9NnlzUp7nfH3X2Rr+NNe+S3o1liLy8o57Tu+x7VY/m+Xfg1/Wzovehb0i1/f1k"
            "w85GPeSrj5LVBGpy6iev/eS19nOE/x+gaS2i"
        ),
    },
    "iq2_xxs": {
        "source": "blk.0.attn_gate.weight",
        "block_bytes": 66,
        "blocks": (
            "eNoBjAFz/h4NhxlMgsOHqpTEeFmmMeZw4V0WDG0g4Cb0pTZTgvOM3r2oaw9MQEyjfPeuF/uVJwiZksqBaESGaYExuAFT"
            "KMEBk0MNCEkDIWIGt8+8CRrWkLA2d7ZC/0m3vOOCDi0WUT6Mgv8VisTSWq625wLzKca0Lpy+i2qXT7x15YoTItH5Rzf0"
            "hmMMII18mY4im+1dZIo77zVm6MUboI8D79T31BB1QJuOy5qNKSG6TWwbwyDmyNH/Kx6iY4bTDpD+o9HE4ACHp1TGvQwM"
            "yGBr0/KEtaKcEJDJ4GDO/xjDFkyPmje7B5AlhYYbWP8QBfPTejik75531FEJNlH+mGkNxp5IHM3KzeC8Sr+o1pkM9A8t"
            "sqJ02ZIZDS8LwnHA040JCXza+pedDYUCN94HQvK5kLGPRoUCkZtRJ4dVbQm0J+VwazuGl3caymqF/qftjjYMkfIiNBP2"
            "4c7htG3cwS21qXiRkebCWsy4lJiFhE8AP6HXO76mrJpLsAYAURyv5J/4toWGKjGmD6mo9CS+VbBguEDGxLA="
        ),
        "expected": (
            "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"
            "AX+DecZ/P39gQw=="
        ),
    },
    "iq2_xs": {
        "source": "blk.2.ssm_out.weight",
        "block_bytes": 74,
        "blocks": (
            "eNoBvAFD/ooQA6AAiCN6lmuU9VeqmJ0A5LvON22xpJNoA/QAWAGcDZaWQKyDoBePECMoCX4R2nrRr2MDSLwulHU1HAI0"
            "CJIA7qhpU211f1h7XxAAzCTiWUXSqgJfCmAn5Z2fsULmOlXie4P+2Hs6K4JXIkNMR9BMRi0ZH7TGIaAzha65Ev8svrLU"
            "1ifnCjik2AR9VzVCZmaZ9mj0FNMeyOaKjppuTNaS4wbCEv+Gcmfqfj5EnRP95Ooudk/83EyYhB7dAMgAhGrkfgw9mb+0"
            "iVCEeyeZATYADOqTF9s0MyQ0hTxVjwANNM9T0JRdV5xbUJAyTRtpVWhT61gGwqYlAPxNcKRXB3CIfKqRCfyjf2kPFAEX"
            "LmoX/nqif6ITk2ALpVceb5N4Tbpntlv4epaLABIJsJWJzbQAtif3Y3IAVhfoXfpC8GUkXYI35fkDxVr/SmoyAAJJMjgb"
            "H0hKjGpgJVmrJd94HzoDLeSgUQN6OSomyKRERK+Iw6ljEmOmdWwqnHp6sDpbIynDWhCVbdJGRFvGr1757kJ49dKyb1ZJ"
            "vCaNS9cwmkYajOIXLCaNWoLNhkaeOkv/ABP8eMYiEiIh58/YfcDzwlE="
        ),
        "expected": (
            "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"
            "/qrzW/DbKMq7yFtqTG/oGeZ9U/1/TYe49w=="
        ),
    },
    "iq2_s": {
        "source": "blk.35.attn_output.weight",
        "block_bytes": 82,
        "blocks": (
            "eNoB7AET/q4M+OCIlAAzLJjmBQAJuLQHGSwziwAmnlIDogF04URRrgOutsGuCoscGISHWf5Z7PnTZCXDydppD389LinF"
            "F+K8HgxMCGQcABGwhFvXRv1Y11zFCvZOQw2g3hUdMb0vDvZmARjtM7DOQZ5kTxxzz/H+AAeVcDR0leZIoJDxcfhrr7Uv"
            "kpvnsBWJYWuM7FOcANKeHe00igDNHJJDAYl2po2NiMp/XwiRcq1apgHtYw6m6g/Aao/ThGi7zgieTN49ZMg3FACEA/WD"
            "VoudZA4dZAYThwSiFspPvOiNjuUxu11ujtJevf5VIolkIiygEQJrv56qq32nzw0KowLwPnr0CswZwQKXCL5JdIKYjwKX"
            "nkrYcesAH00mlyNWkgcISZ9synXVqLyJ7IuNpgFD4XvoXfHtkou3P9FjihCFiMAeQwlxiWa7act/5XglDa3SmBz14UIO"
            "vwsbhiMZ3oIAM4AAgmkAAXRzAAlEAHWHs6+UevSuXXqwriyP2frZUU5JYkM/84aPBbjS5XTjweCoMK6QHAYBoCQ0M2bP"
            "hbc+UA3t0EhwrUstChnOAAkLEQRd7in2AFXGGaCDeNvvEgC4IkeqzF+vBg0/wJbAXSu4r42iDMm+Oqdr4+mcwHeTesEG"
            "wEgAQgMAVoImQ/hldEZUNmfm3b4="
        ),
        "expected": (
            "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"
        ),
    },
}


def packed_blocks(name: str) -> np.ndarray:
    """Packed bytes for ``name`` as ``(n_blocks, block_bytes)`` uint8."""
    entry = _VECTORS[name]
    raw = zlib.decompress(base64.b64decode(entry["blocks"]))
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, entry["block_bytes"]).copy()


def expected_values(name: str) -> np.ndarray:
    """llama.cpp's dequantized output for ``packed_blocks(name)``, as ``(n, 256)``."""
    raw = zlib.decompress(base64.b64decode(_VECTORS[name]["expected"]))
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 256).copy()


def formats() -> list[str]:
    """Every format with captured vectors."""
    return sorted(_VECTORS)
