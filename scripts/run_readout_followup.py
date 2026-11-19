"""Frozen-DKT map-back comparisons with MPS gradients and convex coefficients.

All writes stay in CurriculumKT. Stages separate calibration/fitting from test.
"""
import argparse
import gc
import json
import platform
import runpy
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from evaluate_saved_curriculum_kt import ROOT, SOURCE, OUT as OLD, dump, sha
from curriculum_readouts import table

OUT = ROOT / 'results/curriculum_kt_readout_followup_20261107'
A17 = ROOT / 'results/curriculum_kt_all_methods_20261107/assist2017'
INDUCED = ['monotone_converged','signed_mapped','soft_mapping','full_logits','induced_state_head']
CONTROLS = ['original_calibration','original_state_head']


def save_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    dump(temp, obj)
    temp.replace(path)


def registry(dataset=None, **changes):
    path = OUT / 'run_registry.json'
    x = json.loads(path.read_text())
    (x['datasets'][dataset] if dataset else x).update(changes)
    save_json(path, x)


def runs(dataset):
    if dataset == 'assist2012':
        return json.loads((OLD/'protocol.json').read_text())['runs']
    saved = json.loads((A17/'run_registry.json').read_text())
    assert len(saved)==6 and all(r['state']=='complete' for r in saved.values())
    return [{'kind': r['kind'], 'seed':r['seed'], 'checkpoint':str(A17/key/'best.ckpt'),
             'checkpoint_sha256':r['checkpoint_sha256'],'num_concepts':68 if r['kind']=='original' else 100,
             'embedding_size':256} for key,r in saved.items()]


def load_mapping(dataset):
    return dict(np.load((OLD if dataset=='assist2012' else A17)/'mapping.npz'))


def folder(dataset, run):
    path = OUT/dataset/f"{run['kind']}_seed{run['seed']}"
    path.mkdir(parents=True,exist_ok=True)
    return path


def extract(dataset, run, split):
    dest = folder(dataset,run)
    mark = dest/f'{split}_complete.json'
    if mark.exists():
        info=json.loads(mark.read_text())
        assert info['checkpoint_sha256']==run['checkpoint_sha256']
        return
    start=time.monotonic()
    mapping=load_mapping(dataset)
    if dataset=='assist2012':
        src=OLD/('induced_valid_sequences.npz' if split=='calibration' else 'induced_test_sequences.npz')
        all_data=dict(np.load(src))
        students=np.load(OUT/'assist2012_common_calibration_students.npy' if split=='calibration' else OLD/'shared_student_ids.npy')
        keep=np.isin(all_data['uid'],students)
        d={k:v[keep] for k,v in all_data.items()}
        q=d['questions']; valid=q>=0
        oc=np.full(q.shape,-1,np.int64); oc[valid]=mapping['q_to_original'][q[valid]]
        kc=d['concepts']
        c=oc if run['kind']=='original' else kc
        selected=d['selectmasks'][:,1:]>=0
        rowid=d['row']
        eid=d['timestamps']
    else:
        src=A17/f'{split}_sequences.npz'
        d=dict(np.load(src));q=d['questions'];oc=d['original_concepts'];kc=d['induced_concepts']
        c=oc if run['kind']=='original' else kc
        selected=(c[:,:-1]>=0)&(c[:,1:]>=0)
        rowid=np.arange(len(c));eid=d['event_id']
    mask=(c[:,:-1]>=0)&(c[:,1:]>=0)
    assert np.all(~selected|mask)
    rows,pos=np.where(selected)
    meta={'uid':d['uid'][rows],'row':rowid[rows],'position':pos+1,'event_id':eid[rows,pos+1],
          'label':d['responses'][rows,pos+1],'original_concept':oc[rows,pos+1],
          'induced_concept':kc[rows,pos+1],'question_index':q[rows,pos+1]}
    cp=Path(run['checkpoint']);assert sha(cp)==run['checkpoint_sha256']
    model_cls=runpy.run_path(str(SOURCE/'pykt/models/dkt.py'))['DKT']
    m=model_cls(run['num_concepts'],run['embedding_size'],0.5)
    m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));m.to('mps').eval()
    for p in m.parameters():p.requires_grad_(False)
    assert all(p.device.type=='mps' for p in m.parameters())
    hidden=[];logits=[]
    with torch.inference_mode():
        for lo in range(0,len(c),64):
            sl=slice(lo,lo+64)
            tc=torch.tensor(c[sl,:-1]*mask[sl],device='mps')
            tr=torch.tensor(d['responses'][sl,:-1]*mask[sl],device='mps')
            h,_=m.lstm_layer(m.interaction_emb(tc+m.num_c*tr))
            z=m.out_layer(m.dropout_layer(h))
            if lo==0:
                assert torch.allclose(z.sigmoid(),m(tc,tr),atol=2e-6,rtol=1e-5)
                changed=tr[:1].clone();changed[0,2]=1-changed[0,2]
                assert torch.allclose(m(tc[:1],changed)[:,:2],m(tc[:1],tr[:1])[:,:2],atol=2e-6)
            ix=torch.tensor(selected[sl],device='mps')
            hidden.append(h[ix].cpu().numpy());logits.append(z[ix].cpu().numpy())
    h=np.concatenate(hidden);z=np.concatenate(logits)
    key='original_dkt' if run['kind']=='original' else 'raw_induced'
    target=meta['original_concept'] if run['kind']=='original' else meta['induced_concept']
    meta[key]=expit(z[np.arange(len(z)),target])
    if run['kind']=='induced':
        ps=expit(z)
        for name in ['item_overlap','practice_frequency','equal_children']:
            meta[name]=np.sum(ps*mapping[name][meta['original_concept']],axis=1)
    if dataset=='assist2012' and split=='test':
        ref=np.load(OLD/(f"original_seed{run['seed']}_original_shared_predictions.npz" if run['kind']=='original' else f"induced_seed{run['seed']}_induced_test_predictions.npz"))
        take=np.isin(ref['uid'],students)
        for k in ['uid','row','position','label','original_concept']:
            assert np.array_equal(meta[k],ref[k][take]),k
        assert np.allclose(meta[key],ref[key][take],atol=2e-6,rtol=1e-5)
    np.save(dest/f'{split}_hidden.npy',h)
    np.save(dest/f'{split}_logits.npy',z)
    np.savez_compressed(dest/f'{split}_metadata.npz',**meta)
    weights={'O':m.out_layer.weight.detach().cpu().numpy(),'d':m.out_layer.bias.detach().cpu().numpy()}
    wfile=dest/'output_layer.npz'
    if wfile.exists():
        old=np.load(wfile);assert all(np.array_equal(weights[k],old[k]) for k in weights)
    else: np.savez_compressed(wfile,**weights)
    save_json(mark,{'checkpoint_sha256':run['checkpoint_sha256'],'sequence_source_sha256':sha(src),
        'events':len(z),'students':len(np.unique(meta['uid'])),'device':'mps',
        'parameter_device':'mps:0','hidden_width':h.shape[1],'output_width':z.shape[1],
        'causality_and_forward_identity':True,'seconds':time.monotonic()-start,
        'files':{p.name:sha(p) for p in [dest/f'{split}_hidden.npy',dest/f'{split}_logits.npy',dest/f'{split}_metadata.npz',wfile]}})
    print(json.dumps({'stage':'extract','dataset':dataset,'run':dest.name,'split':split,'events':len(z),'seconds':time.monotonic()-start}),flush=True)
    del m,h,z,hidden,logits;gc.collect();torch.mps.empty_cache()


class ConvexReadout:
    """Actual-coordinate ridge logistic objective; data products run on MPS."""
    def __init__(self,x,c,y,anchor,free,penalty,nonnegative,initial=None):
        self.anchor=np.asarray(anchor,dtype=np.float64)
        self.shape=self.anchor.shape
        self.nc,self.nd=self.shape
        counts=np.bincount(c,minlength=self.nc);positive=np.bincount(c,weights=y,minlength=self.nc)
        self.eligible=(counts>=50)&(positive>=5)&((counts-positive)>=5)
        self.free=np.asarray(free,bool)&self.eligible[:,None]
        self.ix=np.flatnonzero(self.free.ravel());self.bi=np.flatnonzero(self.eligible)
        select=self.eligible[c]
        self.xnp=np.asarray(x[select],np.float32);self.cnp=np.asarray(c[select],np.int64);self.ynp=np.asarray(y[select],np.float64)
        self.np_count=np.maximum(counts,1)
        self.x=torch.tensor(self.xnp,device='mps');self.c=torch.tensor(self.cnp,device='mps')
        self.penalty=np.broadcast_to(np.asarray(penalty,dtype=float),self.shape).copy()
        self.nonnegative=nonnegative
        self.initial_a,self.initial_b=(self.anchor.copy(),np.zeros(self.nc)) if initial is None else (np.asarray(initial[0],float).copy(),np.asarray(initial[1],float).copy())
        self.initial_a[~self.free]=self.anchor[~self.free]
        self.initial_b[~self.eligible]=0
        self.theta=np.r_[self.initial_a.ravel()[self.ix],self.initial_b[self.bi]]
        self.bounds=([(0,None)]*len(self.ix) if nonnegative else [(None,None)]*len(self.ix))+[(None,None)]*len(self.bi)
        self.calls=0;self.start=time.monotonic();self.last=None
        self.counts=counts;self.positive=positive

    def unpack(self,t):
        a=self.anchor.copy();a.ravel()[self.ix]=t[:len(self.ix)]
        b=np.zeros(self.nc);b[self.bi]=t[len(self.ix):]
        return a,b

    def objective(self,t):
        a,b=self.unpack(t)
        with torch.inference_mode():
            aa=torch.tensor(a,dtype=torch.float32,device='mps');bb=torch.tensor(b,dtype=torch.float32,device='mps')
            z=((self.x*aa[self.c]).sum(1)+bb[self.c]).cpu().numpy().astype(np.float64)
            err=(expit(z)-self.ynp)/self.np_count[self.cnp]
            e=torch.tensor(err,dtype=torch.float32,device='mps')
            ga=torch.zeros(self.shape,device='mps').index_add_(0,self.c,self.x*e[:,None]).cpu().numpy().astype(np.float64)
        gb=np.bincount(self.cnp,weights=err,minlength=self.nc)
        delta=a-self.anchor
        loss=np.sum((np.logaddexp(0,z)-self.ynp*z)/self.np_count[self.cnp])
        loss+=np.sum(self.penalty[self.eligible]*delta[self.eligible]**2)+0.001*np.sum(b[self.eligible]**2)
        ga+=2*self.penalty*delta;gb+=0.002*b
        grad=np.r_[ga.ravel()[self.ix],gb[self.bi]]
        self.calls+=1;self.last=(float(loss),grad)
        return float(loss),grad

    def pg(self,t,g):
        g=g.copy()
        if self.nonnegative:
            j=np.arange(len(self.ix));g[j[(t[j]<=1e-8)&(g[j]>0)]]=0
        return float(np.max(np.abs(g)))

    def audit64(self,t):
        # Independent CPU float64 diagnostic of the final fixed coefficients.
        # This verifies the MPS fit; it is not a CPU fitting fallback.
        a,b=self.unpack(t);ga=np.zeros_like(a);gb=np.zeros_like(b);loss=0.
        for c in self.bi:
            ix=self.cnp==c;x=self.xnp[ix].astype(float);y=self.ynp[ix]
            z=x@a[c]+b[c];err=(expit(z)-y)/len(y)
            ga[c]=x.T@err+2*self.penalty[c]*(a[c]-self.anchor[c]);gb[c]=err.sum()+0.002*b[c]
            loss+=np.mean(np.logaddexp(0,z)-y*z)+np.sum(self.penalty[c]*(a[c]-self.anchor[c])**2)+0.001*b[c]**2
        grad=np.r_[ga.ravel()[self.ix],gb[self.bi]]
        return float(loss),self.pg(t,grad)

    def solve(self,label):
        initial_objective,initial_grad=self.objective(self.theta)
        initial_pg=self.pg(self.theta,initial_grad)
        passes=[];theta=self.theta
        for attempt in range(3):
            res=minimize(self.objective,theta,jac=True,method='L-BFGS-B',bounds=self.bounds,
                         options={'maxiter':1500,'maxfun':3000,'ftol':1e-12,'gtol':2e-5,'maxcor':20,'maxls':40})
            objective,gradient=self.objective(res.x);pg=self.pg(res.x,gradient)
            exact_objective,exact_pg=self.audit64(res.x)
            passes.append({'attempt':attempt+1,'success':bool(res.success),'message':str(res.message),
                           'iterations':int(res.nit),'objective':objective,'mps_projected_gradient':pg,
                           'float64_objective':exact_objective,'float64_projected_gradient':exact_pg})
            print(json.dumps({'fit':label,**passes[-1],'seconds':time.monotonic()-self.start}),flush=True)
            theta=res.x
            if exact_pg<=0.0002:break
        a,b=self.unpack(theta)
        info={'label':label,'device':'mps','optimizer_control':'CPU L-BFGS-B; MPS feature products and gradients',
              'independent_final_check':'CPU float64 fixed-coefficient objective/KKT diagnostic',
              'initial_objective_sum_over_concepts':initial_objective,'initial_projected_gradient':initial_pg,
              'final_objective_sum_over_concepts':exact_objective,'projected_gradient_max':exact_pg,
              'converged':bool(exact_pg<=0.0002),'tolerance':0.0002,'function_calls':self.calls,
              'eligible_concepts':int(self.eligible.sum()),'fit_events':len(self.ynp),
              'seconds':time.monotonic()-self.start,'attempts':passes}
        assert exact_objective<=initial_objective+1e-4,(label,'objective increased')
        if self.nonnegative: assert np.min(a[self.free])>=0
        result={'a':a,'b':b,'eligible':self.eligible,'counts':self.counts,'positive':self.positive}
        return result,info


def inputs(method,z,h,c,mapping,w):
    nc=mapping['counts'].shape[0];children,support,weights,_=table(mapping)
    if method in ['monotone_converged','signed_mapped']:
        x=np.take_along_axis(z,children[c],axis=1)
        anchor=weights['practice_frequency'];free=support>0;penalty=0.01
        initial=(anchor,np.zeros(nc))
    elif method in ['soft_mapping','full_logits']:
        x=z;anchor=mapping['practice_frequency'];free=np.ones_like(anchor,bool)
        penalty=np.where(mapping['counts']>0,0.01,0.1) if method=='soft_mapping' else 0.01
        initial=(anchor,np.zeros(nc))
    elif method=='induced_state_head':
        x=h;anchor=mapping['practice_frequency']@w['O'];free=np.ones_like(anchor,bool);penalty=0.01
        initial=(anchor,mapping['practice_frequency']@w['d'])
    elif method=='original_state_head':
        x=h;anchor=w['O'];free=np.ones_like(anchor,bool);penalty=0.01
        initial=(anchor,w['d'])
    elif method=='original_calibration':
        x=z[np.arange(len(z)),c,None];anchor=np.ones((nc,1));free=np.ones_like(anchor,bool);penalty=0.01
        initial=(anchor,np.zeros(nc))
    else:raise ValueError(method)
    return x,anchor,free,penalty,initial


def predict(x,c,params,fallback):
    a,b=params['a'],params['b'];p=np.empty(len(x),float)
    for lo in range(0,len(x),2048):
        sl=slice(lo,lo+2048)
        p[sl]=expit(np.sum(x[sl].astype(float)*a[c[sl]],axis=1)+b[c[sl]])
    missing=~params['eligible'][c];p[missing]=fallback[missing]
    return p


def fit_one(dest,method,x,c,y,anchor,free,penalty,initial):
    file=dest/f'{method}_coefficients.npz';record=file.with_suffix('.json')
    if record.exists():
        info=json.loads(record.read_text())
        assert info['converged'],f'Previously incomplete convergence: {record}'
        assert sha(file)==info['coefficient_sha256']
        return dict(np.load(file)),info
    solver=ConvexReadout(x,c,y,anchor,free,penalty,method in ['monotone_converged','original_calibration'],initial)
    result,info=solver.solve(str(dest.name)+'/'+method)
    np.savez_compressed(file,**result)
    info.update({'coefficient_sha256':sha(file),'test_used_for_fitting':False,'backbone_frozen':True})
    save_json(record,info)
    del solver;gc.collect();torch.mps.empty_cache()
    if not info['converged']:raise RuntimeError(f'Convergence tolerance unmet: {record}')
    return result,info


def fit_run(dataset,run):
    dest=folder(dataset,run);mark=dest/'fit_complete.json'
    if mark.exists():return
    mapping=load_mapping(dataset)
    d=dict(np.load(dest/'calibration_metadata.npz'));c=d['original_concept'];y=d['label']
    z=np.load(dest/'calibration_logits.npy',mmap_mode='r');h=np.load(dest/'calibration_hidden.npy',mmap_mode='r')
    w=dict(np.load(dest/'output_layer.npz'))
    methods=INDUCED if run['kind']=='induced' else CONTROLS
    selected='original_dkt';validation=None
    if run['kind']=='original':
        u=np.unique(d['uid']);rng=np.random.default_rng(20261107);u=rng.permutation(u)
        hold=np.isin(d['uid'],u[:max(1,int(np.ceil(.2*len(u))))]);fit=~hold
        selection_dir=dest/'development_selection';selection_dir.mkdir(exist_ok=True)
        validation={'original_dkt':float(roc_auc_score(y[hold],d['original_dkt'][hold]))}
        for method in methods:
            x,anchor,free,penalty,initial=inputs(method,z,h,c,mapping,w)
            params,_=fit_one(selection_dir,method,x[fit],c[fit],y[fit],anchor,free,penalty,initial)
            p=predict(x[hold],c[hold],params,d['original_dkt'][hold])
            validation[method]=float(roc_auc_score(y[hold],p))
        selected=max(validation,key=validation.get)
        save_json(dest/'baseline_selection.json',{'selected':selected,'heldout_calibration_auc':validation,
                 'holdout_students':len(np.unique(d['uid'][hold])),'fit_students':len(np.unique(d['uid'][fit])),
                 'holdout_student_ids':sorted(int(i) for i in np.unique(d['uid'][hold])),
                 'selection_uses_test':False})
    infos={}
    for method in methods:
        x,anchor,free,penalty,initial=inputs(method,z,h,c,mapping,w)
        _,infos[method]=fit_one(dest,method,x,c,y,anchor,free,penalty,initial)
    save_json(mark,{'methods':methods,'all_converged':all(i['converged'] for i in infos.values()),
                   'checkpoint_sha256':run['checkpoint_sha256'],'selected_original_control':selected,
                   'calibration_metadata_sha256':sha(dest/'calibration_metadata.npz'),
                   'coefficient_hashes':{m:infos[m]['coefficient_sha256'] for m in methods}})


def evaluate_run(dataset,run):
    dest=folder(dataset,run);file=dest/'test_predictions.npz'
    if file.exists():return
    fit=json.loads((dest/'fit_complete.json').read_text());assert fit['all_converged']
    extract(dataset,run,'test')
    mapping=load_mapping(dataset);d=dict(np.load(dest/'test_metadata.npz'))
    z=np.load(dest/'test_logits.npy',mmap_mode='r');h=np.load(dest/'test_hidden.npy',mmap_mode='r')
    w=dict(np.load(dest/'output_layer.npz'));c=d['original_concept']
    for method in fit['methods']:
        cp=dest/f'{method}_coefficients.npz';assert sha(cp)==fit['coefficient_hashes'][method]
        params=dict(np.load(cp));x,*_=inputs(method,z,h,c,mapping,w)
        fallback=d['practice_frequency'] if run['kind']=='induced' else d['original_dkt']
        d[method]=predict(x,c,params,fallback)
        assert sha(cp)==fit['coefficient_hashes'][method]
    if run['kind']=='original':d['original_selected']=d[fit['selected_original_control']].copy()
    np.savez_compressed(file,**d)
    save_json(file.with_suffix('.json'),{'sha256':sha(file),'checkpoint_sha256':run['checkpoint_sha256'],
              'coefficient_hashes_unchanged':True,'fit_complete_before_test':True})
    print(json.dumps({'stage':'test_predictions_saved','dataset':dataset,'run':dest.name}),flush=True)


def old_audit():
    registry(audit_state='running')
    mapping=load_mapping('assist2012');_,support,weights,_=table(mapping)
    for run in runs('assist2012'):
        if run['kind']!='induced':continue
        seed=run['seed'];dest=OUT/'assist2012'/f'old_objective_audit_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'audit.json').exists():continue
        x=np.load(OLD/f'learned_seed{seed}_induced_valid_features.npy',mmap_mode='r')
        d=dict(np.load(OLD/f'induced_seed{seed}_induced_valid_predictions.npz'))
        old=OLD/f'learned_seed{seed}_coefficients.npz';old_hash=sha(old);params=dict(np.load(old))
        result,info=fit_one(dest,'monotone_converged',x,d['original_concept'],d['label'],weights['practice_frequency'],support>0,0.01,(params['a'],params['b']))
        assert np.array_equal(result['eligible'],params['eligible'])
        assert sha(old)==old_hash
        test=np.load(OLD/f'learned_seed{seed}_induced_test_features.npy',mmap_mode='r')
        truth=dict(np.load(OLD/f'induced_seed{seed}_induced_test_predictions.npz'))
        shared=np.isin(truth['uid'],np.load(OLD/'shared_student_ids.npy'))
        pp=predict(test[shared],truth['original_concept'][shared],result,truth['practice_frequency'][shared])
        np.savez_compressed(dest/'test_predictions.npz',uid=truth['uid'][shared],row=truth['row'][shared],position=truth['position'][shared],label=truth['label'][shared],original_concept=truth['original_concept'][shared],audit_monotone=pp)
        save_json(dest/'audit.json',{'seed':seed,'old_coefficients_sha256_unchanged':old_hash,'same_features_and_objective':True,
                   'convergence':info,'test_auc':float(roc_auc_score(truth['label'][shared],pp)),
                   'scope':'Optimization audit on full historical induced calibration; separate from new common-student readout comparison'})
    registry(audit_state='complete',**{'2012_audit':'complete'})


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibration','fit','test','audit']);p.add_argument('--dataset',choices=['assist2012','assist2017'])
    args=p.parse_args();assert torch.backends.mps.is_available(),'MPS required; no CPU fitting fallback'
    torch.set_num_threads(1)
    save_json(OUT/'readout_environment.json',{'python':platform.python_version(),'torch':torch.__version__,'mps':True,'script_sha256':sha(Path(__file__))})
    if args.stage=='audit':old_audit();return
    assert args.dataset
    ds=args.dataset;key={'calibration':'features','fit':'readout_fit','test':'test'}[args.stage]
    registry(ds,**{key:'running'})
    try:
        rr=runs(ds)
        if args.stage=='test':
            assert all(json.loads((folder(ds,r)/'fit_complete.json').read_text())['all_converged'] for r in rr)
        for r in rr:
            if args.stage=='calibration':extract(ds,r,'calibration')
            elif args.stage=='fit':fit_run(ds,r)
            else:evaluate_run(ds,r)
        registry(ds,**{key:'complete'})
    except Exception as exc:
        registry(ds,**{key:'failed','failure':repr(exc)})
        raise


if __name__=='__main__':main()
