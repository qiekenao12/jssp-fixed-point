#!/usr/bin/env python3
"""Source-backed scientific figures and editable CSV tables (no search/refits)."""
import csv,json,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_fp_decision_collection import load,write_new,RESULTS
from scripts.analyze_fp_decision_study import compare

AN=ROOT/'projects/analysis/fp_decision_study';PKG=ROOT/'projects/submission/fp_decision_study'
FIG=PKG/'figures';TABLE=PKG/'tables'
BLUE='#215a7d';GOLD='#b46c19';GREY='#555555'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.titleweight':'normal'})

def write_csv(path,rows):
    if path.exists():raise ValueError(f'Cannot overwrite {path}')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(FIG/f'{name}.{ext}',dpi=250,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def dot(ax,values,labels,scale=1):
    for i,v in enumerate(values):
        ci=v.get('ci95',v.get('hierarchical_bootstrap_95ci'))
        mean=v.get('mean',v.get('mean_error_reduction'))
        ax.errorbar(mean*scale,i,xerr=np.array([[mean-ci[0]],[ci[1]-mean]])*scale,fmt='o',color=BLUE,capsize=3)
    ax.set_yticks(range(len(values)),labels);ax.invert_yaxis();ax.axvline(0,color=GREY,linewidth=.8);ax.grid(axis='x',alpha=.2)

def main():
    FIG.mkdir(parents=True,exist_ok=True);TABLE.mkdir(parents=True,exist_ok=True)
    primary=load(AN/'primary_results.json');tabu=load(AN/'tabu_final_analysis.json');original=load(AN/'original_final_analysis.json')
    fig,ax=plt.subplots(figsize=(10,3.2));ax.axis('off')
    boxes=[(.01,.53,.23,.32,'Saved native state\n+ actual past history'),(.32,.53,.28,.32,'Development-only affine maps\nFP and shared-supervision NoFP'),(.69,.53,.29,.32,'Frozen scores → action choice\nNo branch outcome as input'),(.08,.04,.37,.28,'Detached branches: identical state\nall actions × independent repeats'),(.59,.04,.37,.28,'Instance-level paired evaluation\nstate error / ranking / selected reward')]
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',facecolor='white',edgecolor=BLUE));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
    for start,end in [((.25,.69),(.31,.69)),((.61,.69),(.68,.69)),((.23,.52),(.25,.33)),((.46,.18),(.58,.18)),((.83,.52),(.81,.33))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':GREY})
    ax.text(.5,-.09,'Offline interventions on two independent search implementations; no validated online FP policy.',ha='center',fontsize=9)
    save(fig,'fig1_evaluation_flow')
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.6))
    dyn1=original['dynamics']['comparisons'];dyn2=tabu['dynamics']
    dot(axes[0],dyn1,[f"{r['action']} vs {r['comparator']}" for r in dyn1]);axes[0].set_title('(a) Genetic/memetic stack')
    dot(axes[1],dyn2,[r['contrast'].replace('tabu_','').replace('_error_',' vs ').replace('_minus_conditioned','').replace('CB_BEST_DESCENT','best descent').replace('CB_FIRST_DESCENT','first descent').replace('CB_TABU','tabu') for r in dyn2]);axes[1].set_title('(b) Independent critical-block stack')
    axes[0].set_xlabel('Reference − conditioned error\nweighted 4-D state norm; 95% hierarchical CI')
    axes[1].set_xlabel('Reference − conditioned error\nunweighted 4-D state norm; 95% instance CI')
    fig.tight_layout();save(fig,'fig2_transition_errors')
    fig,ax=plt.subplots(figsize=(9,3.2));dot(ax,primary['primary_family'],['Original: FP E − state/history C','Critical-block: FP D − structural B','Critical-block: FP D − shared-supervision E'],100)
    ax.set_xlabel('Selected offline gain difference (percentage points of initial makespan)\n95% instance-cluster bootstrap intervals, n = 20; no cross-solver effect pooling')
    fig.tight_layout();save(fig,'fig3_primary_decision_contrasts')
    values=defaultdict(list);neutral=[]
    for p in sorted((RESULTS/'original/final').glob('*.json.gz')):
        d=load(p)
        for window,rows in [('atomic',d['records']),('equal480',d['equal_work_records'])]:
            for r in rows:
                if r['generation'] in (40,100,160):values[d['instance'],window,r['action']].append(r['gain'])
        data=d['instance_data'];lb=max(max(map(sum,data['durations'])),max(sum(data['durations'][j][k] for j in range(data['jobs']) for k in range(data['machines']) if data['routes'][j][k]==m) for m in range(data['machines'])))
        neutral.append({'instance':d['instance'],'run':d['run'],'solver':'original','final_makespan':d['main_final_makespan'],'simple_lower_bound':lb,'gap_to_simple_LB_percent':100*(d['main_final_makespan']/lb-1),
                        'neutral_search_seconds':d['main_wall_seconds_excluding_diagnostics']})
    names=sorted({i for i,w,a in values});windows=[]
    for window in ('atomic','equal480'):
        diff={i:float(np.mean(values[i,window,'LS'])-np.mean(values[i,window,'VND'])) for i in names}
        windows.append(compare(diff,'new_original_LS_minus_VND_'+window))
    # Supplementary independent-repeat sample-winner contrasts; no primary-family promotion.
    gains=defaultdict(dict);insts={};variability=defaultdict(list)
    for p in sorted((RESULTS/'tabu/final').glob('*.json.gz')):
        d=load(p);data=d['instance_data'];lb=max(max(map(sum,data['durations'])),max(sum(data['durations'][j][k] for j in range(data['jobs']) for k in range(data['machines']) if data['routes'][j][k]==m) for m in range(data['machines'])))
        neutral.append({'instance':d['instance'],'run':d['run'],'solver':'tabu','final_makespan':d['main_final_makespan'],'simple_lower_bound':lb,'gap_to_simple_LB_percent':100*(d['main_final_makespan']/lb-1),
                        'neutral_search_seconds':sum(r['elapsed_seconds'] for r in d['neutral'])})
        for r in d['records']:
            gains[r['state_id']][r['action'],r['repeat']]=r['gain'];insts[r['state_id']]=r['instance']
    oracle_by_inst=defaultdict(lambda:defaultdict(list));pairwise=[]
    actions=('CB_BEST_DESCENT','CB_FIRST_DESCENT','CB_TABU')
    for sid,g in gains.items():
        matrix=np.array([[g[a,r] for r in range(5)] for a in actions]);fit=matrix[:,:2].mean(1);ev=matrix[:,2:].mean(1);a=int(np.argmax(fit))
        oracle_by_inst[insts[sid]]['fit_advantage'].append(float(fit[a]-fit[2]));oracle_by_inst[insts[sid]]['independent_advantage'].append(float(ev[a]-ev[2]))
        for i in range(3):
            for j in range(i+1,3):
                delta=matrix[i]-matrix[j]
                pairwise.append({'instance':insts[sid],'state_id':sid,'a':actions[i],'b':actions[j],
                    'mean_paired_gain_difference5':float(delta.mean()),'sd_paired_gain_difference5':float(delta.std(ddof=1)),
                    'se_paired_gain_difference5':float(delta.std(ddof=1)/np.sqrt(5))})
    oracle=[compare({i:float(np.mean(g[k])) for i,g in oracle_by_inst.items()},'tabu_oracle_'+k) for k in ('fit_advantage','independent_advantage')]
    historical_window=[{'mean':-.000584228788,'ci95':[-.001061965064,-.000216874638]}, {'mean':.002453269870,'ci95':[.001803808389,.003092391501]}]
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    dot(axes[0],historical_window,['Atomic','480 requests'],100);axes[0].set_title('(a) Historical matched 20 instances')
    dot(axes[1],windows,['Atomic','480 requests'],100);axes[1].set_title('(b) Newly generated 20 instances')
    for ax in axes:ax.set_xlabel('LS − VND normalized gain (percentage points)\n95% instance bootstrap CI; matched states')
    fig.tight_layout();save(fig,'fig4_action_window')
    hist_oracle=[{'mean':.001008754175,'ci95':[.000508872136,.001603850298]}, {'mean':-.000370682368,'ci95':[-.000865546690,.000036881116]}]
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.6))
    dot(axes[0],hist_oracle,['Winner on same 2 repeats','Chosen winner on fresh 5'],100);axes[0].set_title('(a) Historical GA/LS/VND (16 instances)')
    dot(axes[1],oracle,['Winner on same 2 repeats','Chosen winner on fresh 3'],100);axes[1].set_title('(b) Critical-block actions (20 instances)')
    axes[0].set_xlabel('Chosen gain − fixed VND (percentage points)\nOriginal 95% hierarchical intervals')
    axes[1].set_xlabel('Chosen gain − fixed tabu (percentage points)\nNominal 95% instance intervals; secondary')
    fig.tight_layout();save(fig,'fig5_independent_repeat_winners')
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.4))
    for arm in ('B','D','E'):
        i=('B','D','E').index(arm)
        for ai,a in enumerate(actions):
            count=tabu['decision_counts']['gain_'+arm].get(a,0)
            start=sum(tabu['decision_counts']['gain_'+arm].get(b,0) for b in actions[:ai])
            axes[0].barh(i,count/8,left=start/8,color=[BLUE,GOLD,'#b4b4b4'][ai],edgecolor='white',label=a.replace('CB_','') if i==0 else None)
    axes[0].set_yticks(range(3),['Structural B','FP D','Shared-supervision E']);axes[0].invert_yaxis();axes[0].set_xlim(0,100);axes[0].set_xlabel('Decisions (%) of 800 opportunities');axes[0].legend(fontsize=7,loc='lower left',bbox_to_anchor=(0,-.65),ncol=1)
    per=list(csv.DictReader((AN/'tabu_instance_metrics.csv').open()));scale_values=[]
    for size in ('6x6','10x5','10x10','15x10'):
        v=[float(r['gain_D_value'])-float(r['gain_B_value']) for r in per if '_'+size+'_' in r['instance']];scale_values.append(float(np.mean(v)))
    axes[1].barh(range(4),np.array(scale_values)*100,color=BLUE);axes[1].set_yticks(range(4),['6×6','10×5','10×10','15×10']);axes[1].invert_yaxis();axes[1].axvline(0,color=GREY,lw=.8);axes[1].set_xlabel('FP D − B gain (percentage points)\nDescriptive size means, five instances each')
    fig.tight_layout();save(fig,'fig6_choices_and_scope')
    write_csv(TABLE/'new_neutral_trajectory_outcomes.csv',neutral);write_csv(TABLE/'tabu_paired_action_differences.csv',pairwise)
    write_csv(TABLE/'primary_contrasts.csv',[{'contrast':r['contrast'],'mean':r['mean'],'ci_low':r['ci95'][0],'ci_high':r['ci95'][1],'exact_p':r['p'],'holm_p':r['holm_p'],'wins':r['win_tie_loss'][0],'ties':r['win_tie_loss'][1],'losses':r['win_tie_loss'][2]} for r in primary['primary_family']])
    description={'new_original_window_contrasts':windows,'tabu_oracle_advantage':oracle,
        'size_D_minus_B':dict(zip(['6x6','10x5','10x10','15x10'],scale_values)),
        'sources':['projects/analysis/fp_decision_study/primary_results.json','projects/analysis/fp_decision_study/tabu_final_analysis.json','projects/analysis/fp_decision_study/original_final_analysis.json','projects/docs/fp_decision_history_evidence.md','results/fp_decision_study'],
        'intervals':'Main primary20000instance bootstrap; inherited original dynamics10000hierarchical; historical oracle original frozen hierarchical; window descriptive20000instance. Units explicitly distinct. Supplementary exploratory contrasts are not primary or Holm-confirmed claims.',
        'no_online_curve':'All curves and intervals concern recorded transitions or branch gains. Neutral outcomes are not an FP/NoFP online policy comparison.'}
    write_new(AN/'figure_source_data.json',description)
    print('Created six figures in PNG/PDF/editable SVG and three editable CSV tables')

if __name__=='__main__':main()
