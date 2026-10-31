"""Regenerate the offline, line-linked source browser from an explicit allowlist.

No network, package dependency, repository scan, credentials or runtime files.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "humanoidverse/train.py",
    "humanoidverse/agents/base.py",
    "humanoidverse/agents/base_model.py",
    "humanoidverse/agents/fb/agent.py",
    "humanoidverse/agents/fb/model.py",
    "humanoidverse/agents/fb_cpr/agent.py",
    "humanoidverse/agents/fb_cpr/model.py",
    "humanoidverse/agents/fb_cpr_aux/agent.py",
    "humanoidverse/agents/fb_cpr_aux/model.py",
    "humanoidverse/agents/nn_models.py",
    "humanoidverse/agents/nn_filters.py",
    "humanoidverse/agents/misc/zbuffer.py",
    "humanoidverse/agents/envs/humanoidverse_isaac.py",
    "humanoidverse/agents/buffers/trajectory.py",
    "humanoidverse/agents/buffers/transition.py",
    "humanoidverse/agents/evaluations/humanoidverse_isaac.py",
    "humanoidverse/envs/legged_base_task/legged_robot_base.py",
    "humanoidverse/envs/legged_robot_motions/legged_robot_motions.py",
    "humanoidverse/envs/g1_env_helper/bench/reward_eval_hv.py",
    "humanoidverse/utils/helpers.py",
    "humanoidverse/config/obs/bfm_zero_obs.yaml",
    "humanoidverse/config/exp/bfm_zero/bfm_zero.yaml",
    "humanoidverse/config/robot/g1/g1_29dof_hard_waist.yaml",
    "humanoidverse/config/rewards/reward_bfm_zero.yaml",
    "humanoidverse/config/domain_rand/domain_rand.yaml",
    "humanoidverse/config/simulator/isaacsim.yaml",
    "humanoidverse/tracking_inference.py",
    "humanoidverse/goal_inference.py",
    "humanoidverse/reward_inference.py",
    "experiments/train.py",
    "experiments/evaluate.py",
    "experiments/preflight.py",
    "experiments/log_videos.py",
    "cluster/pyproject.toml",
    "cluster/job.sh",
    "cluster/cluv.sh",
    "apptainer/bfm-zero.def",
]


def main():
    sources, symbols = {}, {}
    for path in FILES:
        source = (ROOT / path).read_text()
        sources[path] = source
        if path.endswith(".py"):
            def walk(node, parents=()):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = (*parents, child.name)
                        symbols[f"{path}:{'.'.join(name)}"] = child.lineno
                        walk(child, name)
                    else:
                        walk(child, parents)
            walk(ast.parse(source))
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    index = {"commit": commit, "built": datetime.now(timezone.utc).strftime("%Y-%m-%d UTC"), "symbols": symbols}
    (ROOT / "docs/source-index.js").write_text("window.BFM_SOURCE_INDEX=" + json.dumps(index, separators=(",", ":")) + ";\n")
    data = json.dumps(sources).replace("<", "\\u003c").replace("&", "\\u0026")
    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BFM-Zero · Local source browser</title><link rel="stylesheet" href="guide.css"></head><body class="source-page"><main><header><h1>Read the source.</h1><div class="source-tools"><a href="index.html">← Field guide</a><label for="source-file">File</label><select id="source-file"></select><a id="raw-source">Raw file ↗</a></div><div class="source-caption" id="caption"></div></header><pre class="source-code" id="code" aria-label="Source code"></pre></main><script>const sources=SOURCE_DATA;
const select=document.getElementById('source-file');for(const path of Object.keys(sources)){const option=document.createElement('option');option.value=path;option.textContent=path;select.append(option);}
function show(){const match=decodeURIComponent(location.hash.slice(1)).match(/^(.*):L(\\d+)$/);const path=match&&sources[match[1]]!==undefined?match[1]:Object.keys(sources)[0];const line=match?Number(match[2]):1;select.value=path;document.getElementById('caption').textContent=path+' · local source snapshot · line '+line;document.getElementById('raw-source').href='../'+path;const pre=document.getElementById('code');pre.replaceChildren();sources[path].split('\\n').forEach((text,i)=>{const span=document.createElement('span');span.className='source-line'+(i+1===line?' selected':'');span.id='L'+(i+1);const a=document.createElement('a');a.className='line-number';a.href='#'+encodeURIComponent(path)+':L'+(i+1);a.textContent=String(i+1);span.append(a,document.createTextNode(text));pre.append(span);});requestAnimationFrame(()=>document.getElementById('L'+line)?.scrollIntoView({block:'center'}));}
select.addEventListener('change',()=>location.hash=encodeURIComponent(select.value)+':L1');window.addEventListener('hashchange',show);show();</script></body></html>'''
    (ROOT / "docs/source.html").write_text(html.replace("SOURCE_DATA", data))
    print(f"Built offline source browser: {len(sources)} files, {len(symbols)} symbols.")


if __name__ == "__main__":
    main()
