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

"""Self-contained AIPerf comparison report."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import median
from typing import Any

from .schema import BenchmarkResult

__all__ = ["write_aiperf_report"]


def _comparison_points(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    """Aggregate repeated measurements for comparison plots."""

    groups: dict[tuple[str, str, str, str, str, int], list[BenchmarkResult]] = {}
    for result in results:
        key = (
            result.checkpoint_dir,
            result.solution_id,
            result.profile_id,
            result.topology_id,
            result.workload_id,
            result.concurrency,
        )
        groups.setdefault(key, []).append(result)

    points = []
    for (
        checkpoint_dir,
        solution_id,
        profile_id,
        topology_id,
        workload_id,
        concurrency,
    ), rows in groups.items():
        metric_names = set.intersection(*(set(row.metrics) for row in rows))
        series_id = json.dumps(
            [checkpoint_dir, solution_id, profile_id, topology_id], separators=(",", ":")
        )
        points.append(
            {
                "checkpoint_dir": checkpoint_dir,
                "solution_id": solution_id,
                "profile_id": profile_id,
                "topology_id": topology_id,
                "series_id": series_id,
                "series_label": f"{solution_id} [{profile_id} / {topology_id}]",
                "workload_id": workload_id,
                "workload": rows[0].workload,
                "concurrency": concurrency,
                "repetitions": len(rows),
                "metrics": {
                    name: median(float(row.metrics[name]) for row in rows)
                    for name in sorted(metric_names)
                },
            }
        )
    return sorted(
        points,
        key=lambda point: (
            point["workload_id"],
            point["concurrency"],
            point["solution_id"],
            point["profile_id"],
            point["topology_id"],
        ),
    )


def write_aiperf_report(results: list[BenchmarkResult], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "aiperf_results.json"
    csv_path = output_dir / "aiperf_results.csv"
    html_path = output_dir / "aiperf_report.html"
    rows = [
        result.model_dump(mode="json") if hasattr(result, "model_dump") else vars(result)
        for result in results
    ]
    comparison_points = _comparison_points(results)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    metric_names = sorted({key for result in results for key in result.metrics})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "architecture_id",
                "solution_id",
                "profile_id",
                "topology_id",
                "workload_id",
                "checkpoint_dir",
                "concurrency",
                "repetition",
                "serialized_size_bytes",
                "parameter_count",
                "result_fingerprint",
                "failures",
                *metric_names,
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "architecture_id": result.architecture_id,
                    "solution_id": getattr(result, "solution_id", "unknown"),
                    "profile_id": getattr(result, "profile_id", "unknown"),
                    "topology_id": getattr(result, "topology_id", "unknown"),
                    "workload_id": getattr(result, "workload_id", "unknown"),
                    "checkpoint_dir": result.checkpoint_dir,
                    "concurrency": result.concurrency,
                    "repetition": getattr(result, "repetition", 0),
                    "serialized_size_bytes": getattr(result, "checkpoint_identity", {}).get(
                        "serialized_size_bytes"
                    ),
                    "parameter_count": getattr(result, "checkpoint_identity", {}).get(
                        "parameter_count"
                    ),
                    "result_fingerprint": getattr(result, "result_fingerprint", "unknown"),
                    "failures": result.failures,
                    **result.metrics,
                }
            )
    table_rows = []
    for result in results:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(result.solution_id)}</td>"
            f"<td>{result.workload.get('image_batch_size', 0)}</td>"
            f"<td>{result.concurrency}</td>"
            f"<td>{getattr(result, 'repetition', 0)}</td>"
            f"<td>{result.metrics.get('output_token_throughput', float('nan')):.6g}</td>"
            f"<td>{result.metrics.get('ttft_mean_ms', float('nan')):.6g}</td>"
            f"<td>{result.metrics.get('tpot_mean_ms', float('nan')):.6g}</td>"
            f"<td>{result.metrics.get('request_latency_mean_ms', float('nan')):.6g}</td>"
            f"<td>{result.failures}</td>"
            "</tr>"
        )
    plot_data = json.dumps(comparison_points).replace("</", "<\\/")
    plot_metrics = json.dumps(
        sorted({key for point in comparison_points for key in point["metrics"]})
    ).replace("</", "<\\/")
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Puzzletron AIPerf Report</title>"
        "<style>"
        ":root{font-family:Inter,system-ui,sans-serif;color:#172033;background:#f6f8fc}"
        "body{margin:0 auto;padding:2rem;max-width:1400px}h1,h2{margin:.2rem 0 1rem}"
        ".panel{background:white;border:1px solid #dce2ee;border-radius:12px;padding:1rem 1.2rem;"
        "box-shadow:0 2px 8px #1720330c;margin:1rem 0}.controls{display:grid;grid-template-columns:minmax(180px,.3fr) minmax(220px,.35fr) 1fr;gap:1rem}"
        "label{font-weight:600}select{display:block;width:100%;margin-top:.35rem;padding:.5rem;border:1px solid #b7c1d3;border-radius:6px}"
        ".series{display:flex;flex-wrap:wrap;gap:.45rem .85rem;margin-top:.45rem}.series label{font-weight:400;white-space:nowrap}"
        ".buttons{margin-top:.55rem;display:flex;gap:.4rem}button{border:1px solid #a9b5c9;background:#f7f9fc;border-radius:5px;padding:.3rem .65rem;cursor:pointer}"
        "svg.plot{width:100%;height:auto;min-height:460px}.chart-note{color:#536078;font-size:.9rem}"
        "table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d7deea;padding:.4rem;text-align:right}"
        "th{background:#eef2f8;position:sticky;top:0}th:first-child,td:first-child{text-align:left}.table-wrap{overflow:auto;max-height:620px}"
        "@media(max-width:800px){.controls{grid-template-columns:1fr}}"
        "</style>"
        "<h1>Puzzletron AIPerf comparison</h1>"
        "<section class='panel'><h2>Serving metrics</h2><div class='controls'><label>Metric<select id='metric'></select></label>"
        "<label>Workload<select id='workload'></select></label>"
        "<div><label>Solutions</label><div id='series' class='series'></div>"
        "<div class='buttons'><button id='all' type='button'>Select all</button><button id='none' type='button'>Clear</button></div></div></div>"
        "<p id='chart-note' class='chart-note'></p><svg id='chart' class='plot' viewBox='0 0 1100 560' role='img' aria-label='Median AIPerf metric by concurrency'></svg></section>"
        "<section class='panel'><h2>Raw measurements</h2><div class='table-wrap'><table><thead><tr><th>solution</th><th>images</th><th>concurrency</th><th>repetition</th><th>output tok/s</th>"
        "<th>TTFT mean ms</th><th>TPOT mean ms</th><th>latency mean ms</th><th>failures</th>"
        "</tr></thead><tbody>" + "".join(table_rows) + "</tbody></table></div></section>"
        f"<script id='plot-data' type='application/json'>{plot_data}</script>"
        f"<script id='metric-data' type='application/json'>{plot_metrics}</script>"
        "<script>"
        "const rows=JSON.parse(document.getElementById('plot-data').textContent);"
        "const metrics=JSON.parse(document.getElementById('metric-data').textContent);"
        "const metricSelect=document.getElementById('metric'),workloadSelect=document.getElementById('workload'),seriesBox=document.getElementById('series'),svg=document.getElementById('chart');"
        "const NS='http://www.w3.org/2000/svg',colors=['#2563eb','#dc2626','#059669','#7c3aed','#ea580c','#0891b2','#be123c','#4d7c0f','#9333ea','#0f766e','#b45309','#475569'];"
        "const labelOf=id=>rows.find(r=>r.series_id===id)?.series_label||id;"
        "const workloadLabel=r=>{const w=r.workload||{},images=Number(w.image_batch_size||0);return `${images?images+' image'+(images===1?'':'s'):'text only'} | ${w.input_tokens||'?'} in / ${w.output_tokens||'?'} out`};"
        "const seriesIds=[...new Set(rows.map(r=>r.series_id))].sort((a,b)=>labelOf(a).localeCompare(labelOf(b),undefined,{numeric:true}));"
        "const workloads=[...new Map(rows.map(r=>[r.workload_id,workloadLabel(r)])).entries()].sort((a,b)=>a[1].localeCompare(b[1],undefined,{numeric:true}));"
        "metrics.forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=m.replaceAll('_',' ');metricSelect.appendChild(o)});"
        "metricSelect.value=metrics.includes('output_token_throughput')?'output_token_throughput':metrics[0];"
        "workloads.forEach(([id,label])=>{const o=document.createElement('option');o.value=id;o.textContent=label;workloadSelect.appendChild(o)});"
        "seriesIds.forEach((id,i)=>{const l=document.createElement('label'),c=document.createElement('input');c.type='checkbox';c.value=id;c.checked=true;c.dataset.color=colors[i%colors.length];c.addEventListener('change',renderTrend);l.append(c,document.createTextNode(' '+labelOf(id)));l.style.color=colors[i%colors.length];seriesBox.appendChild(l)});"
        "metricSelect.addEventListener('change',renderTrend);workloadSelect.addEventListener('change',renderTrend);document.getElementById('all').onclick=()=>{seriesBox.querySelectorAll('input').forEach(x=>x.checked=true);renderTrend()};"
        "document.getElementById('none').onclick=()=>{seriesBox.querySelectorAll('input').forEach(x=>x.checked=false);renderTrend()};"
        "function el(name,attrs={},text=''){const n=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}"
        "function selected(){return new Set([...seriesBox.querySelectorAll('input:checked')].map(x=>x.value))}"
        "function renderTrend(){svg.replaceChildren();const metric=metricSelect.value,chosen=selected();"
        "const data=rows.filter(r=>chosen.has(r.series_id)&&r.workload_id===workloadSelect.value&&Number.isFinite(Number(r.metrics[metric])));"
        "const note=document.getElementById('chart-note');note.textContent=`Median ${metric.replaceAll('_',' ')} across measured repetitions. Hover a point for its exact value.`;"
        "if(!data.length){svg.appendChild(el('text',{x:550,y:280,'text-anchor':'middle',fill:'#64748b'},'Select at least one solution.'));return}"
        "const W=1100,H=560,M={l:100,r:190,t:35,b:75},xs=[...new Set(data.map(r=>Number(r.concurrency)))].sort((a,b)=>a-b),ys=data.map(r=>Number(r.metrics[metric]));"
        "let ymin=Math.min(...ys),ymax=Math.max(...ys);const pad=(ymax-ymin||Math.abs(ymax)||1)*.08;ymin-=pad;ymax+=pad;const xmin=Math.min(...xs),xmax=Math.max(...xs);"
        "const X=x=>M.l+(xmax===xmin?.5:(x-xmin)/(xmax-xmin))*(W-M.l-M.r),Y=y=>M.t+(ymax-y)/(ymax-ymin)*(H-M.t-M.b);"
        "for(let i=0;i<=5;i++){const y=ymin+(ymax-ymin)*i/5,py=Y(y);svg.appendChild(el('line',{x1:M.l,y1:py,x2:W-M.r,y2:py,stroke:'#e5e9f1'}));svg.appendChild(el('text',{x:M.l-10,y:py+4,'text-anchor':'end',fill:'#5b667a','font-size':12},Number(y.toPrecision(5)).toString()))}"
        "xs.forEach(x=>{const px=X(x);svg.appendChild(el('line',{x1:px,y1:M.t,x2:px,y2:H-M.b,stroke:'#f0f2f7'}));svg.appendChild(el('text',{x:px,y:H-M.b+25,'text-anchor':'middle',fill:'#5b667a','font-size':13},String(x)))});"
        "svg.appendChild(el('line',{x1:M.l,y1:M.t,x2:M.l,y2:H-M.b,stroke:'#758198'}));svg.appendChild(el('line',{x1:M.l,y1:H-M.b,x2:W-M.r,y2:H-M.b,stroke:'#758198'}));"
        "svg.appendChild(el('text',{x:(M.l+W-M.r)/2,y:H-20,'text-anchor':'middle',fill:'#29344a','font-size':14},'concurrency'));"
        "const yl=el('text',{x:20,y:(M.t+H-M.b)/2,'text-anchor':'middle',fill:'#29344a','font-size':14,transform:`rotate(-90 20 ${(M.t+H-M.b)/2})`},metric.replaceAll('_',' '));svg.appendChild(yl);"
        "seriesIds.filter(id=>chosen.has(id)).forEach((id,i)=>{const points=data.filter(r=>r.series_id===id).sort((a,b)=>a.concurrency-b.concurrency),color=colors[seriesIds.indexOf(id)%colors.length];if(!points.length)return;"
        "svg.appendChild(el('path',{d:points.map((r,j)=>`${j?'L':'M'} ${X(Number(r.concurrency))} ${Y(Number(r.metrics[metric]))}`).join(' '),fill:'none',stroke:color,'stroke-width':2.5}));"
        "points.forEach(r=>{const c=el('circle',{cx:X(Number(r.concurrency)),cy:Y(Number(r.metrics[metric])),r:4.5,fill:color,stroke:'white','stroke-width':1.5});c.appendChild(el('title',{},`${labelOf(id)} | concurrency ${r.concurrency} | ${metric} = ${Number(r.metrics[metric]).toPrecision(7)}`));svg.appendChild(c)});"
        "const ly=M.t+18*i;svg.appendChild(el('line',{x1:W-M.r+18,y1:ly,x2:W-M.r+42,y2:ly,stroke:color,'stroke-width':3}));svg.appendChild(el('text',{x:W-M.r+48,y:ly+4,fill:'#344057','font-size':12},labelOf(id)))});"
        "}renderTrend();"
        "</script>",
        encoding="utf-8",
    )
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate a self-contained AIPerf report")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    payload = json.loads((args.output_dir / "aiperf_results.json").read_text())
    write_aiperf_report([BenchmarkResult(**row) for row in payload], args.output_dir)
