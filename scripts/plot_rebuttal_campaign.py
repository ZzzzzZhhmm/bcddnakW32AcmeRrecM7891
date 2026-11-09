"""Plot verified fixed-case outcomes, keeping pending roles visible."""
from pathlib import Path
import argparse,json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verified-pairs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.verified_pairs.read_text());rows=data['pairs']
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    a.output_dir.mkdir(parents=True,exist_ok=True)
    values=np.array([[r['roles'][role]['success'] if r['roles'][role] else -1 for role in ('full','candidate_null')] for r in rows])
    fig,ax=plt.subplots(figsize=(5.4,max(3,0.38*len(rows))))
    ax.imshow(values,cmap=ListedColormap(['#d8d8d8','#d98c71','#5ca98c']),norm=BoundaryNorm([-1.5,-.5,.5,1.5],3),aspect='auto')
    for i,row in enumerate(values):
        for j,v in enumerate(row):ax.text(j,i,'pending' if v<0 else str(v),ha='center',va='center',fontsize=9)
    ax.set_xticks([0,1],['Full','Candidate-null']);ax.set_yticks(range(len(rows)),[str(r['episode_id']) for r in rows]);ax.set_ylabel('Fixed episode index')
    ax.set_title(data['task'].replace('_',' ')+'\n'+str(data['summary']['complete_pairs'])+'/'+str(len(rows))+' pairs complete')
    fig.text(.5,.01,'1 = success; 0 = completed failure; pending = no admitted outcome',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(a.output_dir/'paired_outcomes.svg');fig.savefig(a.output_dir/'paired_outcomes.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
