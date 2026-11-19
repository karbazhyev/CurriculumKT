"""Read-only process inventory without printing commands or credential arguments."""
import json
from pathlib import Path
import re
import shlex
import subprocess


def inspect():
    rows=[]
    raw=subprocess.check_output(['ps','-axo','pid=,ppid=,etime=,command='],text=True)
    for line in raw.splitlines():
        fields=line.strip().split(None,3)
        if len(fields)!=4:continue
        pid,parent,elapsed,command=fields
        try:args=shlex.split(command)
        except ValueError:continue
        if not args:continue
        executable=Path(args[0]).name
        if not any(s in executable.lower() for s in ['python','llama-server','caffeinate']):continue
        scripts=[Path(a).name for a in args[1:] if a.endswith('.py')]
        script=scripts[0] if scripts else None
        own_project=(str(Path(__file__).resolve().parents[1])+'/' in command or
                     (script and re.match(r'(run|check|extract|verify|summarize)_(robustkt|simplekt|akt|sakt|dkvmn|dkt|sparsekt|cskt|dtransformer|ukt|kqn|deep_irt|skvmn|saint|atkt)',script)))
        # Resource inventory only: no other-project paths/configuration/queues
        # are read, and no process is signalled or controlled.
        if not own_project and executable.lower()=='caffeinate':continue
        # Only explicit allowlisted non-sensitive fields leave this function.
        mode=None
        if '--mode' in args:
            i=args.index('--mode')
            if i+1<len(args) and args[i+1] in ['generate','comet','metricx']:mode=args[i+1]
        gpu_layers=None
        if '-ngl' in args:
            i=args.index('-ngl')
            if i+1<len(args) and re.fullmatch(r'-?\d+',args[i+1]):gpu_layers=int(args[i+1])
        rows.append({'pid':int(pid),'parent_pid':int(parent),'elapsed':elapsed,
                     'executable':executable,'script':script if own_project else None,
                     'mode':mode,'gpu_layers':gpu_layers,
                     'project':'CurriculumKT' if own_project else 'external'})
    return rows


if __name__=='__main__':
    print(json.dumps(inspect(),indent=2))
