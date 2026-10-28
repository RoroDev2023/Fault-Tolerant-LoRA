#!/usr/bin/env python3
"""Run base and LoRA GEMMs on SCALE-Sim v3, collect the requested 3 scenarios.

SIMULATOR RESULTS: original dense MAC schedule, and low-rank A/B MAC schedules.
NOT YET SIMULATED: index lookup cycles, codebook reconstruction, actual faults,
LoRA output addition, non-GEMM kernels, true branch overlap, shared-layer buffers.
Cluster+LoRA and Cluster+fault+LoRA therefore have the same GEMM schedule by design.
"""
import argparse, csv, json, sys
from pathlib import Path


def report(out):
    candidates=list(Path(out).rglob('COMPUTE_REPORT.csv'))
    if len(candidates)!=1:
        raise FileNotFoundError(f'Expected one COMPUTE_REPORT.csv under {out}, found {candidates}')
    with candidates[0].open(newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        row['Total Cycles'] = int(row[' Total Cycles'].strip()) if ' Total Cycles' in row else int(row['Total Cycles'])
        pref=next(k for k in row if 'incl. prefetch' in k)
        row['Prefetch-inclusive cycles']=int(row[pref])
    return rows


def execute(cfg, workload, output, layout):
    from scalesim.scale_sim import scalesim
    output.mkdir(parents=True,exist_ok=True)
    # save_disk_space=True disables giant detailed SRAM/DRAM traces while keeping CSV summaries.
    sim=scalesim(save_disk_space=True, verbose=False,config=str(cfg),
                 topology=str(workload),layout=str(layout),input_type_gemm=True)
    sim.run_scale(top_path=str(output))
    return report(output)


def tally(rows):
    return sum(int(x['Total Cycles']) for x in rows),sum(int(x['Prefetch-inclusive cycles']) for x in rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workloads',default='hardware_sim/scalesim/workloads')
    p.add_argument('--config',default='../scale-sim-v3/configs/scale.cfg')
    p.add_argument('--layout',default='../scale-sim-v3/layouts/conv_nets/test.csv')
    p.add_argument('--out',default='results/hardware_sim/scalesim')
    args=p.parse_args()
    w=Path(args.workloads);out=Path(args.out)
    original=execute(Path(args.config),w/'original_gemm.csv',out/'base',Path(args.layout))
    lora=execute(Path(args.config),w/'lora_gemm.csv',out/'adapters',Path(args.layout))
    n_base,n_base_pref=tally(original);n_lora,n_lora_pref=tally(lora)
    cases=[('original',n_base,n_base_pref),
           ('clustered+lora',n_base+n_lora,n_base_pref+n_lora_pref),
           ('clustered+faults+lora',n_base+n_lora,n_base_pref+n_lora_pref)]
    out.mkdir(parents=True,exist_ok=True)
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['scenario','serial_sum_total_cycles','serial_sum_prefetch_inclusive_cycles',
                         'overhead_percent_vs_original','scope'])
        for name,total,pref in cases:
            writer.writerow([name,total,pref,round(100*(total/n_base-1),4),
                 'dense GEMM + LoRA GEMM only; NOT compressed-memory or fault cycles'])
    base_names=[r['name'] for r in json.loads((w/'manifest.json').read_text())['layers']]
    layer_rows=[]
    for j,name in enumerate(base_names):
        lora_a,lora_b = lora[2*j],lora[2*j+1]
        layer_rows.append([name,original[j]['Total Cycles'],lora_a['Total Cycles'],lora_b['Total Cycles'],
                           original[j]['Total Cycles']+lora_a['Total Cycles']+lora_b['Total Cycles']])
    with (out/'layer_comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['layer','base_cycles','lora_A_cycles','lora_B_cycles','serial_total_cycles']);writer.writerows(layer_rows)
    print('\nGEMM-only experiment summary')
    for name,total,pref in cases:
        print(f'{name:26s} cycles={total:,} incl_prefetch_serial_sum={pref:,} overhead={100*(total/n_base-1):.2f}%')
    print('Saved',out/'comparison.csv','and',out/'layer_comparison.csv')
    print('WARNING: not yet a codebook/fault timing simulation; layer summations ignore graph overlap.')

if __name__=='__main__':main()
