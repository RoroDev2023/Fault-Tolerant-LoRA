#!/usr/bin/env python3
"""Create a USER-bandwidth copy of a SCALE-Sim v3 config without altering original."""
import argparse
import configparser
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,default=Path('../scale-sim-v3/configs/scale.cfg'))
    p.add_argument('--out',type=Path,default=Path('hardware_sim/scalesim/scale_32x32_user_bw.cfg'))
    p.add_argument('--bandwidth',type=int,default=10,help='SCALE-Sim bandwidth in data elements per cycle (not automatically bytes/cycle)')
    args=p.parse_args()
    if args.bandwidth<1:
        p.error('bandwidth must be positive')
    c=configparser.ConfigParser()
    if not c.read(args.base):
        p.error(f'Cannot read {args.base}')
    if 'run_presets' not in c or 'architecture_presets' not in c:
        p.error('Unexpected SCALE-Sim config sections')
    c['run_presets']['InterfaceBandwidth']='USER'
    c['architecture_presets']['Bandwidth']=str(args.bandwidth)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w') as f:c.write(f)
    print('Wrote',args.out,'mode USER, bandwidth',args.bandwidth,'elements/cycle')

if __name__=='__main__':main()
