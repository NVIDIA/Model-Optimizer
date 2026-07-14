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

"""Generate self-contained interactive reports for Puzzletron sweep diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .sweep_data import (
    AXIS_SPECS,
    SweepRecord,
    automatic_anchors,
    curve_warnings,
    load_replace_block_records,
    load_vllm_records,
    metric_direction,
    observed_axes,
    records_for_anchor,
    sample_layers,
    write_records_csv,
)

__all__ = ["generate_replace_block_report", "generate_vllm_stats_report"]


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _plotly_javascript() -> str:
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as error:
        raise RuntimeError(
            "Generating Puzzletron diagnostic HTML requires Plotly. "
            "Install the 'puzzletron' optional dependencies or `pip install plotly`."
        ) from error
    return get_plotlyjs()


def _curve_analyses(
    records: list[SweepRecord],
    axes: list[str],
    *,
    metric: str,
    expected: str,
    tolerance: float,
    anchor_count: int,
    grouping: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for axis_id in axes:
        for anchor_index, anchor in enumerate(
            automatic_anchors(records, axis_id, count=anchor_count)
        ):
            matching = records_for_anchor(records, axis_id, anchor)
            groups: dict[Any, list[SweepRecord]] = defaultdict(list)
            for record in matching:
                key = record.layer_idx if grouping == "layer" else record.profile_id
                groups[key].append(record)
            for group, curve in sorted(groups.items(), key=lambda item: str(item[0])):
                points = [
                    (record.axes[axis_id], record.metrics[metric])
                    for record in curve
                    if metric in record.metrics and axis_id in record.axes
                ]
                analysis = curve_warnings(points, expected=expected, relative_tolerance=tolerance)
                results.append(
                    {
                        "axis": axis_id,
                        "anchor_index": anchor_index,
                        grouping: group,
                        "metric": metric,
                        "anchor": anchor,
                        **analysis,
                    }
                )
    return results


def _payload_records(records: Iterable[SweepRecord]) -> list[dict[str, Any]]:
    return [record.to_dict() for record in records]


def _render_html(payload: dict[str, Any], title: str) -> str:
    # Prevent source paths/config strings from prematurely terminating the data script.
    data_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    plotly_js = _plotly_javascript()
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: light; --ink:#18212f; --muted:#607086; --line:#d9e0e8; --panel:#f7f9fb; --accent:#2f6fed; --warn:#9a5b00; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:#fff; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; }
header { padding:22px 28px 14px; border-bottom:1px solid var(--line); }
h1 { margin:0 0 4px; font-size:24px; } h2 { font-size:18px; } h3 { font-size:15px; }
.muted { color:var(--muted); }.small { font-size:12px; }
.tabs { display:flex; gap:4px; padding:12px 28px 0; }
button,select { font:inherit; } button { cursor:pointer; }
.tab { border:0; border-bottom:3px solid transparent; padding:9px 15px; background:transparent; }
.tab.active { color:var(--accent); border-color:var(--accent); }
.page { display:none; padding:20px 28px 36px; }.page.active { display:block; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:14px 0; }
.card,.panel { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff; }
.card .value { font-size:22px; font-weight:650; }
.controls { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin-bottom:12px; }
label { display:flex; flex-direction:column; gap:5px; color:var(--muted); font-size:12px; }
select { min-height:34px; padding:5px; border:1px solid #bfc9d5; border-radius:5px; background:white; color:var(--ink); }
select[multiple] { min-height:90px; }
.fixed { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:8px; }
.config-summary { padding:10px 12px; background:var(--panel); border-radius:5px; }
.layer-table { max-height:260px; overflow:auto; border:1px solid var(--line); }
.layer-table input { width:16px; height:16px; vertical-align:middle; cursor:pointer; }
.plot { min-height:460px; }.mini-plot { min-height:320px; }
.toolbar { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
.action { color:white; background:var(--accent); border:0; border-radius:5px; padding:8px 12px; }
.secondary { color:var(--ink); background:white; border:1px solid #bfc9d5; }
.warnings { color:var(--warn); }.warnings li { margin:3px 0; }
pre { max-height:250px; overflow:auto; padding:10px; background:var(--panel); border-radius:5px; white-space:pre-wrap; overflow-wrap:anywhere; }
.all-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:12px; }
table { border-collapse:collapse; width:100%; font-size:12px; } th,td { text-align:left; border-bottom:1px solid var(--line); padding:6px; vertical-align:top; } td { max-width:360px; overflow-wrap:anywhere; }
.table-wrap { overflow:auto; max-height:70vh; border:1px solid var(--line); }
@media (max-width:650px) { .page,header { padding-left:14px;padding-right:14px; }.tabs { padding-left:14px; }.all-grid { grid-template-columns:1fr; } }
</style>
<script>__PLOTLY__</script>
</head>
<body>
<header><h1>__TITLE__</h1><div id="subtitle" class="muted"></div></header>
<nav class="tabs"><button class="tab active" data-tab="overview">Overview</button><button class="tab" data-tab="explorer">Explorer</button><button class="tab" data-tab="data">Data</button></nav>
<main>
<section id="overview" class="page active">
  <div id="cards" class="cards"></div>
  <div class="panel"><h2>All collected candidates</h2><div id="landscapeInfo" class="muted"></div><div id="landscape" class="plot"></div></div>
  <div class="panel"><h2>Automatic-view warnings</h2><div id="summaryWarnings"></div></div>
  <div class="toolbar"><button id="renderAll" class="action">Render all sweeps</button></div>
  <div id="allSweeps" class="all-grid"></div>
</section>
<section id="explorer" class="page">
  <div class="controls">
    <label>Metric<select id="metric"></select></label>
    <label>Swept axis<select id="axis"></select></label>
    <label>Compatible configuration<select id="config"></select></label>
  </div>
  <div class="panel"><h3>Selected complete configuration</h3><div id="configSummary" class="config-summary"></div></div>
  <div id="layerPanel" class="panel" hidden><h3>Layer plot controls</h3><div class="muted small">All compatible layers are enabled by default. Toggle a row to show or hide that layer in the plot.</div><div class="layer-table"><table id="layerTable"></table></div></div>
  <div class="toolbar"><button id="download" class="action">Download filtered CSV</button></div>
  <div id="plot" class="plot"></div>
  <div class="panel"><h3>Sanity warnings</h3><div id="warnings"></div><h3>Plotted configuration and provenance</h3><pre id="details"></pre></div>
</section>
<section id="data" class="page"><div class="toolbar"><button id="downloadAll" class="action">Download all records CSV</button></div><div id="tableInfo" class="muted"></div><div class="table-wrap"><table id="table"></table></div></section>
</main>
<script id="reportData" type="application/json">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('reportData').textContent);
const $=id=>document.getElementById(id);
const canonical=v=>JSON.stringify(v,Object.keys(v||{}).sort());
const selected=el=>Array.from(el.selectedOptions).map(o=>o.value);
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let currentConfigs=[], currentConfig=null, currentRows=[], visibleLayers=new Set();

function option(value,label,chosen=false){const o=document.createElement('option');o.value=String(value);o.textContent=label;o.selected=chosen;return o;}
function setOptions(el,items,chosen){el.innerHTML='';items.forEach(([v,l])=>el.appendChild(option(v,l,(chosen||[]).includes(String(v)))));}
function axisIgnored(axis){const s=D.axisSpecs[axis];return new Set([`${s.kind}.${s.field}`,...s.coupled_fields.map(f=>`${s.kind}.${f}`),`${s.kind}.no_op`]);}
function isNoop(r,axis){const s=D.axisSpecs[axis];return r.fields[`${s.kind}.no_op`]===true;}
function anchorFields(r,axis){const ignored=axisIgnored(axis),out={};Object.keys(r.fields).sort().forEach(k=>{if(!ignored.has(k))out[k]=r.fields[k];});return out;}
function same(a,b){return JSON.stringify(a)===JSON.stringify(b);}
function matchAnchor(r,axis,a){
  if(!(axis in r.axes))return false;
  if(isNoop(r,axis)){const p=`${D.axisSpecs[axis].kind}.`;return Object.entries(a).every(([k,v])=>k.startsWith(p)||same(r.fields[k],v));}
  return same(anchorFields(r,axis),a);
}
function metricRows(metric=$('metric').value){return D.records.filter(r=>finite(r.metrics[metric]));}
function capacity(a){return Object.values(a).reduce((n,v)=>n+(typeof v==='number'?v:0),0);}
function configChoices(rows,axis){
  const groups=new Map();
  rows.filter(r=>axis in r.axes&&!isNoop(r,axis)).forEach(r=>{
    const a=anchorFields(r,axis),key=JSON.stringify(a);if(!groups.has(key))groups.set(key,{a,by:new Map()});
    const trace=D.kind==='replace'?r.layer_idx:r.profile_id;
    if(!groups.get(key).by.has(trace))groups.get(key).by.set(trace,new Set());groups.get(key).by.get(trace).add(r.axes[axis]);
  });
  const choices=Array.from(groups.values()).map(g=>({
    anchor:g.a,
    traces:Array.from(g.by.entries()).filter(([,xs])=>xs.size>=2).map(([trace])=>trace).sort((a,b)=>D.kind==='replace'?a-b:String(a).localeCompare(String(b))),
  })).filter(g=>g.traces.length);
  choices.sort((x,y)=>y.traces.length-x.traces.length||capacity(y.anchor)-capacity(x.anchor)||JSON.stringify(x.anchor).localeCompare(JSON.stringify(y.anchor)));
  return choices;
}
function valueLabel(v){return typeof v==='number'&&Number.isInteger(v)?String(v):String(v);}
function configLabel(a){
  const parts=[],used=new Set();
  if(a['ffn.no_op']===true){parts.push('FFN (no-op)');used.add('ffn.no_op');}
  else if(Object.prototype.hasOwnProperty.call(a,'ffn.intermediate_size')){parts.push(`FFN (intermediate=${valueLabel(a['ffn.intermediate_size'])})`);used.add('ffn.intermediate_size');used.add('ffn.no_op');}
  if(a['attention.no_op']===true){parts.push('Attention (no-op)');used.add('attention.no_op');}
  else {
    const fields=[['attention.num_kv_heads','KV groups'],['attention.q_heads_per_group','Q heads/group'],['attention.qk_head_dim','QK head dim']].filter(([key])=>Object.prototype.hasOwnProperty.call(a,key));
    if(fields.length)parts.push(`Attention (${fields.map(([key,name])=>`${name}=${valueLabel(a[key])}`).join(', ')})`);
    fields.forEach(([key])=>used.add(key));used.add('attention.num_query_heads');used.add('attention.no_op');
  }
  if(a['mamba.no_op']===true){parts.push('GDN (no-op)');used.add('mamba.no_op');}
  else {
    const fields=[['mamba.gdn_key_groups','key groups'],['mamba.gdn_value_heads_per_group','value heads/group'],['mamba.state_dim','key dim'],['mamba.head_dim','value dim']].filter(([key])=>Object.prototype.hasOwnProperty.call(a,key));
    if(fields.length)parts.push(`GDN (${fields.map(([key,name])=>`${name}=${valueLabel(a[key])}`).join(', ')})`);
    fields.forEach(([key])=>used.add(key));['mamba.num_groups','mamba.num_heads','mamba.conv_kernel_size','mamba.no_op'].forEach(key=>used.add(key));
  }
  if(a['moe.no_op']===true){parts.push('MoE (no-op)');used.add('moe.no_op');}
  else {
    const fields=[['moe.num_experts','experts'],['moe.expert_intermediate_size','expert intermediate'],['moe.shared_expert_intermediate_size','shared intermediate'],['moe.top_k','top-k'],['moe.latent_dim','latent dim']].filter(([key])=>Object.prototype.hasOwnProperty.call(a,key));
    if(fields.length)parts.push(`MoE (${fields.map(([key,name])=>`${name}=${valueLabel(a[key])}`).join(', ')})`);fields.forEach(([key])=>used.add(key));used.add('moe.no_op');
  }
  Object.keys(a).sort().filter(key=>!used.has(key)&&!key.endsWith('.no_op')).forEach(key=>parts.push(`${key}=${valueLabel(a[key])}`));
  return parts.join(' + ')||'No additional fixed dimensions';
}
function rebuildConfigs(preserve=false){
  const prior=preserve&&currentConfig?JSON.stringify(currentConfig.anchor):null;
  currentConfigs=configChoices(metricRows(),$('axis').value);
  setOptions($('config'),currentConfigs.map((c,i)=>[i,configLabel(c.anchor)]),[]);
  let index=prior?currentConfigs.findIndex(c=>JSON.stringify(c.anchor)===prior):-1;if(index<0)index=0;$('config').value=String(index);
  currentConfig=currentConfigs[index]||null;
  visibleLayers=new Set(D.kind==='replace'&&currentConfig?currentConfig.traces:[]);
  renderConfigSummary();renderLayerTable();
}
function renderConfigSummary(){
  if(!currentConfig){$('configSummary').textContent='No compatible complete configuration has two measured points for this metric and axis.';return;}
  const traceText=D.kind==='replace'?`${currentConfig.traces.length} compatible layers`:`${currentConfig.traces.length} compatible runtime profile${currentConfig.traces.length===1?'':'s'}`;
  $('configSummary').innerHTML=`<strong>${esc(configLabel(currentConfig.anchor))}</strong><div class="muted small">${esc(traceText)} · only the selected swept axis varies</div><pre>${esc(JSON.stringify(currentConfig.anchor,null,2))}</pre>`;
}
function renderLayerTable(){
  if(D.kind!=='replace'){ $('layerPanel').hidden=true;return; }
  $('layerPanel').hidden=false;const layers=currentConfig?currentConfig.traces:[];const axis=$('axis').value,metric=$('metric').value;
  const rows=metricRows(metric);$('layerTable').innerHTML=`<thead><tr><th>Plot</th><th>Layer</th><th>Measured axis values</th></tr></thead><tbody>${layers.map(layer=>{const values=Array.from(new Set(rows.filter(r=>r.layer_idx===layer&&matchAnchor(r,axis,currentConfig.anchor)).map(r=>r.axes[axis]))).sort((a,b)=>a-b);return `<tr><td><input class="layer-toggle" type="checkbox" data-layer="${layer}" ${visibleLayers.has(layer)?'checked':''}></td><td>Layer ${layer}</td><td>${esc(values.join(', '))}</td></tr>`;}).join('')}</tbody>`;
  document.querySelectorAll('.layer-toggle').forEach(box=>box.addEventListener('change',()=>{const layer=Number(box.dataset.layer);if(box.checked)visibleLayers.add(layer);else visibleLayers.delete(layer);draw(false);}));
}
function groupedCurve(rows,axis,metric,group){
  const filtered=rows.filter(r=>(D.kind==='replace'?r.layer_idx:r.profile_id)===group&&matchAnchor(r,axis,currentConfig.anchor));
  const by=new Map();filtered.forEach(r=>{const x=r.axes[axis];if(!by.has(x))by.set(x,[]);by.get(x).push(r);});return by;
}
function ranks(values){const order=values.map((v,i)=>[v,i]).sort((a,b)=>a[0]-b[0]),out=new Array(values.length);let i=0;while(i<order.length){let j=i+1;while(j<order.length&&order[j][0]===order[i][0])j++;const rank=(i+j-1)/2;for(let k=i;k<j;k++)out[order[k][1]]=rank;i=j;}return out;}
function spearman(xs,ys){if(xs.length<3||new Set(xs).size<2||new Set(ys).size<2)return null;const xr=ranks(xs),yr=ranks(ys),xm=xr.reduce((a,b)=>a+b,0)/xr.length,ym=yr.reduce((a,b)=>a+b,0)/yr.length;let n=0,xd=0,yd=0;for(let i=0;i<xr.length;i++){n+=(xr[i]-xm)*(yr[i]-ym);xd+=(xr[i]-xm)**2;yd+=(yr[i]-ym)**2;}return xd&&yd?n/Math.sqrt(xd*yd):null;}
function curveAnalysis(xs,ys,expected,tol,hasNoop){
  const warnings=[];const vals=ys.filter(finite);if(vals.length<2)warnings.push('fewer than two measured points');if(!hasNoop)warnings.push('no-op measurement is missing');
  if(vals.length>=2){const scale=Math.max(...vals.map(Math.abs),1e-12);if(Math.max(...vals)-Math.min(...vals)<=tol*scale)warnings.push('curve is flat within tolerance');
    for(let i=1;i<ys.length;i++){if(!finite(ys[i-1])||!finite(ys[i]))continue;const t=tol*Math.max(Math.abs(ys[i-1]),Math.abs(ys[i]),1e-12);const bad=expected==='higher'?ys[i]<ys[i-1]-t:ys[i]>ys[i-1]+t;if(bad){warnings.push('direction violation');break;}}
    if(hasNoop&&finite(ys[0])){const rest=ys.slice(1).filter(finite);const bad=D.kind==='vllm'?rest.some(y=>ys[0]>y+tol*Math.max(Math.abs(y),1e-12)):expected==='lower'?rest.some(y=>ys[0]<y-tol*Math.max(Math.abs(y),1e-12)):rest.some(y=>ys[0]>y+tol*Math.max(Math.abs(y),1e-12));if(bad)warnings.push('no-op is not the expected endpoint');}
  }const pairs=xs.map((x,i)=>[x,ys[i]]).filter(([,y])=>finite(y));const raw=spearman(pairs.map(p=>p[0]),pairs.map(p=>p[1]));return {warnings,correlation:raw===null?null:(expected==='higher'?raw:-raw)};
}
function makePlot(rows,axis,metric,anchor,div,compact=false,groupsOverride=null){
  const saved=currentConfig;currentConfig={anchor,traces:groupsOverride||[]};const groups=groupsOverride||(D.kind==='replace'?Array.from(visibleLayers).sort((a,b)=>a-b):(saved?saved.traces:[]));currentConfig.traces=groups;
  const maps=groups.map(g=>[g,groupedCurve(rows,axis,metric,g)]);const union=Array.from(new Set(maps.flatMap(([,m])=>Array.from(m.keys())))).sort((a,b)=>a-b);const notes=[],correlations=[];
  const traces=maps.map(([group,map])=>{
    const y=union.map(x=>{const rs=map.get(x)||[];return rs.length?rs.reduce((n,r)=>n+r.metrics[metric],0)/rs.length:null;});
    const custom=union.map(x=>(map.get(x)||[]).map(r=>r.source_path).join('\n'));
    const missing=union.filter((x,i)=>!finite(y[i]));if(missing.length)notes.push(`${D.kind==='replace'?'layer':'profile'} ${group}: missing x-values ${missing.join(', ')}`);
    const duplicate=Array.from(map.entries()).some(([,rs])=>new Set(rs.map(r=>r.metrics[metric])).size>1);if(duplicate)notes.push(`${group}: inconsistent duplicate measurements`);
    const expected=D.kind==='vllm'?'higher':D.metricDirections[metric],analysis=curveAnalysis(union,y,expected,D.tolerance,map.has(0));analysis.warnings.forEach(w=>notes.push(`${D.kind==='replace'?'layer':'profile'} ${group}: ${w}`));if(analysis.correlation!==null)correlations.push({trace:group,direction_corrected_spearman:analysis.correlation});
    return {x:union,y,customdata:custom,type:'scatter',mode:'lines+markers',connectgaps:false,name:D.kind==='replace'?`Layer ${group}`:D.profileLabels[group],hovertemplate:`${D.axisSpecs[axis].label}: %{x}<br>${metric}: %{y}<br>%{customdata}<extra>%{fullData.name}</extra>`};
  });
  Plotly.react(div,traces,{title:compact?`${D.axisSpecs[axis].label}`:`${metric} vs ${D.axisSpecs[axis].label}`,xaxis:{title:D.axisSpecs[axis].label},yaxis:{title:metric},margin:{t:compact?45:60,r:25,b:60,l:70},legend:{orientation:'h'},hovermode:'closest'},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:`${D.kind}_${axis}_${metric}`}});
  currentConfig=saved;return {notes:Array.from(new Set(notes)),correlations,union,maps};
}
function draw(refreshLayerTable=true){
  const axis=$('axis').value,metric=$('metric').value,rows=metricRows(metric);if(!currentConfig){$('plot').innerHTML='<p>No compatible complete configuration has at least two measured points for this axis.</p>';$('warnings').textContent='Choose another metric or swept axis.';currentRows=[];renderTable();return;}
  const groups=D.kind==='replace'?Array.from(visibleLayers).sort((a,b)=>a-b):currentConfig.traces;
  const result=makePlot(rows,axis,metric,currentConfig.anchor,$('plot'),false,groups);const groupSet=new Set(groups);currentRows=rows.filter(r=>matchAnchor(r,axis,currentConfig.anchor)&&groupSet.has(D.kind==='replace'?r.layer_idx:r.profile_id));
  $('warnings').innerHTML=result.notes.length?`<ul class="warnings">${result.notes.map(w=>`<li>${esc(w)}</li>`).join('')}</ul>`:'<span class="muted">No sanity warnings for this view.</span>';
  const detail={axis,metric,configuration:configLabel(currentConfig.anchor),fixed_config:currentConfig.anchor,visible_layers:D.kind==='replace'?groups:undefined,compatible_profiles:D.kind==='vllm'?groups:undefined,direction_corrected_spearman:result.correlations,source_artifacts:Array.from(new Set(currentRows.map(r=>r.source_path)))};$('details').textContent=JSON.stringify(detail,null,2);if(refreshLayerTable)renderLayerTable();renderTable();
}
function renderTable(){
  const rows=D.records,shown=rows.slice(0,1000),cols=['layer/profile',...D.axes,...D.metrics,'source artifact','block config'];
  $('tableInfo').textContent=`${rows.length} total normalized records; showing ${shown.length}. Download all records for the complete table.`;
  $('table').innerHTML=`<thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${shown.map(r=>`<tr><td>${esc((D.kind==='replace'?r.layer_idx:r.profile_id)??'')}</td>${D.axes.map(axis=>`<td>${esc(r.axes[axis]??'')}</td>`).join('')}${D.metrics.map(metric=>`<td>${esc(r.metrics[metric]??'')}</td>`).join('')}<td>${esc(r.source_path)}</td><td>${esc(JSON.stringify(r.block_config))}</td></tr>`).join('')}</tbody>`;
}
function csvCell(v){const s=typeof v==='string'?v:JSON.stringify(v);return `"${String(s??'').replaceAll('"','""')}"`;}
function downloadRows(rows,axes,metrics,filename){const head=['layer_idx','profile_id',...axes,...metrics,'source_path','block_config','profile','provenance'];const lines=[head.map(csvCell).join(',')];rows.forEach(r=>lines.push([r.layer_idx,r.profile_id,...axes.map(axis=>r.axes[axis]),...metrics.map(metric=>r.metrics[metric]),r.source_path,r.block_config,r.profile,r.provenance].map(csvCell).join(',')));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));a.download=filename;a.click();URL.revokeObjectURL(a.href);}
function downloadCsv(){const axis=$('axis').value,metric=$('metric').value;downloadRows(currentRows,[axis],[metric],`${D.kind}_${axis}_${metric}.csv`);}
function downloadAllCsv(){downloadRows(D.records,D.axes,D.metrics,`${D.kind}_all_records.csv`);}
function blockFamily(r){if(Object.prototype.hasOwnProperty.call(r.fields,'attention.no_op'))return 'attention';if(Object.prototype.hasOwnProperty.call(r.fields,'mamba.no_op'))return 'mamba / GDN';if(Object.prototype.hasOwnProperty.call(r.fields,'moe.no_op'))return 'MoE';return 'other';}
function renderLandscape(){
  const metric=$('metric').value,rows=metricRows(metric),families=new Map();rows.forEach(r=>{const family=blockFamily(r);if(!families.has(family))families.set(family,[]);families.get(family).push(r);});
  let traces,title,xTitle;
  if(D.kind==='vllm'){
    const ordered=rows.slice().sort((a,b)=>a.metrics[metric]-b.metrics[metric]),rank=new Map(ordered.map((r,i)=>[r,i+1]));
    traces=Array.from(families.entries()).map(([family,items])=>({x:items.map(r=>rank.get(r)),y:items.map(r=>r.metrics[metric]),customdata:items.map(r=>[r.source_path,JSON.stringify(r.block_config)]),type:'scatter',mode:'markers',name:family,marker:{size:9,opacity:.75},hovertemplate:'Candidate rank: %{x}<br>'+metric+': %{y}<br>%{customdata[1]}<br>%{customdata[0]}<extra>%{fullData.name}</extra>'}));
    title=`All ${rows.length} runtime candidates`;xTitle=`Candidate rank (ascending ${metric})`;
  }else{
    traces=Array.from(families.entries()).map(([family,items])=>({x:items.map(r=>r.layer_idx),y:items.map(r=>r.metrics[metric]),customdata:items.map(r=>[r.source_path,JSON.stringify(r.block_config)]),type:'scatter',mode:'markers',name:family,marker:{size:8,opacity:.58},hovertemplate:'Layer: %{x}<br>'+metric+': %{y}<br>%{customdata[1]}<br>%{customdata[0]}<extra>%{fullData.name}</extra>'}));
    title=`All ${rows.length} replace-one-block candidates`;xTitle='Layer';
  }
  $('landscapeInfo').textContent='Every normalized record with the selected metric is included; marginal Explorer plots below remain fixed-configuration slices.';
  Plotly.react($('landscape'),traces,{title,xaxis:{title:xTitle},yaxis:{title:metric},margin:{t:55,r:25,b:60,l:75},legend:{orientation:'h'},hovermode:'closest'},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:`${D.kind}_all_candidates_${metric}`}});
}
function renderAll(){
  const box=$('allSweeps');box.innerHTML='';const rows=metricRows(),metric=$('metric').value;D.axes.forEach(axis=>{configChoices(rows,axis).slice(0,D.anchorCount).forEach(config=>{const panel=document.createElement('div');panel.className='panel';const meta=document.createElement('div');meta.className='small muted';meta.textContent=`${configLabel(config.anchor)} — ${config.traces.length} ${D.kind==='replace'?'compatible layers':'runtime profiles'}`;const div=document.createElement('div');div.className='mini-plot';panel.append(meta,div);box.appendChild(panel);makePlot(rows,axis,metric,config.anchor,div,true,config.traces);});});if(!box.children.length)box.textContent='No compatible automatic sweeps for the current metric.';
}
function init(){
  document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.tab,.page').forEach(e=>e.classList.remove('active'));b.classList.add('active');$(b.dataset.tab).classList.add('active');if(b.dataset.tab==='data')renderTable();}));
  $('subtitle').textContent=D.subtitle;setOptions($('metric'),D.metrics.map(x=>[x,x]),[D.defaultMetric]);setOptions($('axis'),D.axes.map(x=>[x,D.axisSpecs[x].label]),[D.axes[0]]);
  const warningCount=D.summaryWarnings.length+D.ingestionWarnings.length;const cards=[['Records',D.records.length],['Axes',D.axes.length],[D.kind==='replace'?'Layers':'Runtime profiles',D.kind==='replace'?D.layers.length:D.profiles.length],['Metrics',D.metrics.length],['Automatic warnings',warningCount]];$('cards').innerHTML=cards.map(([k,v])=>`<div class="card"><div class="muted">${esc(k)}</div><div class="value">${esc(v)}</div></div>`).join('');
  const warningItems=[...D.ingestionWarnings.map(w=>`<li>Ingestion / ${esc(w.metric)}: ${esc(w.warning)} — ${esc(w.source_path)}</li>`),...D.summaryWarnings.map(w=>`<li>${esc(w.axis)} / ${esc(w.layer??w.profile)}: ${esc(w.warnings.join(', '))}</li>`)];$('summaryWarnings').innerHTML=warningItems.length?`<ul class="warnings">${warningItems.slice(0,100).join('')}</ul>${warningItems.length>100?'<p>Showing first 100 warnings.</p>':''}`:'<span class="muted">No warnings in automatic views.</span>';
  $('metric').addEventListener('change',()=>{rebuildConfigs(true);draw();renderLandscape();});$('axis').addEventListener('change',()=>{rebuildConfigs();draw();});
  $('config').addEventListener('change',()=>{currentConfig=currentConfigs[Number($('config').value)]||null;visibleLayers=new Set(D.kind==='replace'&&currentConfig?currentConfig.traces:[]);renderConfigSummary();renderLayerTable();draw(false);});$('download').addEventListener('click',downloadCsv);$('downloadAll').addEventListener('click',downloadAllCsv);$('renderAll').addEventListener('click',renderAll);
  rebuildConfigs();draw();renderLandscape();renderTable();
}
init();
</script>
</body></html>"""
    return (
        template.replace("__TITLE__", title)
        .replace("__PLOTLY__", plotly_js)
        .replace("__DATA__", data_json)
    )


def _common_payload(
    records: list[SweepRecord],
    *,
    kind: str,
    default_metric: str,
    tolerance: float,
    anchor_count: int,
    warnings: list[dict[str, Any]],
    ingestion_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    axes = observed_axes(records)
    metrics = sorted({metric for record in records for metric in record.metrics})
    if not axes:
        raise ValueError("No axis has at least two observed values")
    if default_metric not in metrics:
        default_metric = metrics[0]
    return {
        "kind": kind,
        "records": _payload_records(records),
        "axes": axes,
        "axisSpecs": {
            axis: {
                "label": AXIS_SPECS[axis].label,
                "kind": AXIS_SPECS[axis].kind,
                "field": AXIS_SPECS[axis].field,
                "coupled_fields": list(AXIS_SPECS[axis].coupled_fields),
            }
            for axis in axes
        },
        "metrics": metrics,
        "defaultMetric": default_metric,
        "metricDirections": {metric: metric_direction(metric) for metric in metrics},
        "tolerance": tolerance,
        "anchorCount": anchor_count,
        "summaryWarnings": warnings,
        "ingestionWarnings": ingestion_warnings,
    }


def generate_vllm_stats_report(
    puzzle_dir: str | Path,
    *,
    stats_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    granularity: str = "block",
    anchor_count: int = 3,
    trend_relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Generate the offline vLLM block-runtime diagnostic report and sidecars."""

    puzzle_dir = Path(puzzle_dir).resolve()
    stats_path = Path(stats_path or puzzle_dir / "subblock_stats.json").resolve()
    if granularity not in {"block", "subblock"}:
        raise ValueError(f"Unsupported vLLM-stat granularity: {granularity}")
    output_dir = Path(output_dir or puzzle_dir / "artifacts" / "vllm_stats").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ingestion_warnings: list[dict[str, Any]] = []
    records = load_vllm_records(stats_path, puzzle_dir=puzzle_dir, issues=ingestion_warnings)
    axes = observed_axes(records)
    analyses = _curve_analyses(
        records,
        axes,
        metric="runtime_ms",
        expected="higher",
        tolerance=trend_relative_tolerance,
        anchor_count=anchor_count,
        grouping="profile",
    )
    payload = _common_payload(
        records,
        kind="vllm",
        default_metric="runtime_ms",
        tolerance=trend_relative_tolerance,
        anchor_count=anchor_count,
        warnings=[analysis for analysis in analyses if analysis["warnings"]],
        ingestion_warnings=ingestion_warnings,
    )
    profiles = sorted({record.profile_id for record in records if record.profile_id})
    profile_values = {record.profile_id: record.profile for record in records if record.profile_id}
    payload.update(
        {
            "profiles": profiles,
            "profileLabels": {
                profile: f"{profile}: {json.dumps(profile_values[profile], sort_keys=True)}"
                for profile in profiles
            },
            "subtitle": f"Subblock runtime, memory, cache, and FLOP metrics from {stats_path}",
        }
    )
    html_path = output_dir / "vllm_stats_sanity.html"
    html_path.write_text(_render_html(payload, "Puzzletron vLLM Stats Sanity"), encoding="utf-8")
    csv_path = write_records_csv(output_dir / "normalized_records.csv", records)
    warnings_path = _write_json(
        output_dir / "warnings.json", {"ingestion": ingestion_warnings, "curves": analyses}
    )
    summary = {
        "kind": "vllm_stats",
        "granularity": granularity,
        "inputs": [str(stats_path)],
        "outputs": [str(html_path), str(csv_path), str(warnings_path)],
        "record_count": len(records),
        "axes": axes,
        "profiles": profiles,
        "metrics": payload["metrics"],
        "warning_count": len(ingestion_warnings)
        + sum(bool(analysis["warnings"]) for analysis in analyses),
        "analysis_count": len(analyses),
    }
    summary_path = output_dir / "summary.json"
    summary["outputs"].append(str(summary_path))
    _write_json(summary_path, summary)
    return summary


def generate_replace_block_report(
    puzzle_dir: str | Path,
    *,
    scores_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    granularity: str = "block",
    default_metric: str = "normalized_mse_loss_hidden_states",
    default_layer_count: int = 5,
    anchor_count: int = 3,
    trend_relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Generate an offline replace-one block/subblock score diagnostic and sidecars."""

    puzzle_dir = Path(puzzle_dir).resolve()
    if granularity not in {"block", "subblock"}:
        raise ValueError(f"Unsupported scoring granularity: {granularity}")
    unit_label = "subblock" if granularity == "subblock" else "block"
    scores_dir = Path(
        scores_dir
        or puzzle_dir
        / (
            "single_subblock_replacement_solutions--validation"
            if granularity == "subblock"
            else "single_sequence_replacement_solutions--validation"
        )
    ).resolve()
    output_dir = Path(output_dir or puzzle_dir / "artifacts" / "replacement_scoring").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ingestion_warnings: list[dict[str, Any]] = []
    records = load_replace_block_records(
        scores_dir, puzzle_dir=puzzle_dir, issues=ingestion_warnings
    )
    axes = observed_axes(records)
    metrics = sorted({metric for record in records for metric in record.metrics})
    if default_metric not in metrics:
        default_metric = metrics[0]
    analyses = _curve_analyses(
        records,
        axes,
        metric=default_metric,
        expected=metric_direction(default_metric),
        tolerance=trend_relative_tolerance,
        anchor_count=anchor_count,
        grouping="layer",
    )
    payload = _common_payload(
        records,
        kind="replace",
        default_metric=default_metric,
        tolerance=trend_relative_tolerance,
        anchor_count=anchor_count,
        warnings=[analysis for analysis in analyses if analysis["warnings"]],
        ingestion_warnings=ingestion_warnings,
    )
    layers = sorted({record.layer_idx for record in records if record.layer_idx is not None})
    payload.update(
        {
            "layers": layers,
            "sampleLayers": sample_layers(layers, default_layer_count),
            "profiles": [],
            "profileLabels": {},
            "subtitle": f"Replace-one-{unit_label} scores from {scores_dir}",
        }
    )
    html_path = output_dir / "replace_block_sanity.html"
    html_path.write_text(
        _render_html(payload, f"Puzzletron Replace-One-{unit_label.title()} Sanity"),
        encoding="utf-8",
    )
    csv_path = write_records_csv(output_dir / "normalized_records.csv", records)
    warnings_path = _write_json(
        output_dir / "warnings.json", {"ingestion": ingestion_warnings, "curves": analyses}
    )
    summary = {
        "kind": "replacement_scoring",
        "granularity": granularity,
        "inputs": [str(scores_dir)],
        "outputs": [str(html_path), str(csv_path), str(warnings_path)],
        "record_count": len(records),
        "axes": axes,
        "layers": layers,
        "metrics": payload["metrics"],
        "warning_count": len(ingestion_warnings)
        + sum(bool(analysis["warnings"]) for analysis in analyses),
        "analysis_count": len(analyses),
    }
    summary_path = output_dir / "summary.json"
    summary["outputs"].append(str(summary_path))
    _write_json(summary_path, summary)
    return summary
