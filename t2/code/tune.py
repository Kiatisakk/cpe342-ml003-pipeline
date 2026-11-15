import sys,time,json;sys.path.insert(0,'.');from nb import *
import optuna;optuna.logging.set_verbosity(optuna.logging.WARNING)
WEAK=['ranked_participation_rate','competitive_rank','watches_esports','team_play_percentage','collection_progress','achievement_completion_rate']
# cache per-feature log-liks per (feature, nbins, sm, alpha) per fold to make trials fast
cache={}
def LL(c,fo,nb_,sm,al):
    key=(c,fo,nb_,round(sm,2),round(al,2))
    if key not in cache:
        a,b=folds[fo];cache[key]=loglik(c,a,b,nbins=nb_,sm=sm,alpha=al)
    return cache[key]
def scores(p):
    L=np.zeros((len(y),K))
    for fo,(a,b) in enumerate(folds):
        s=np.log(np.bincount(y[a],minlength=K)/len(a))+np.zeros((len(b),K))
        for c in FEATS:
            if c in WEAK and not p[f'use_{c}']: continue
            nb_=p['nb_cont'];sm=p['sm'];al=p['alpha']
            w=p['w_disc'] if c in DISC else 1.0
            s=s+w*LL(c,fo,nb_ if c in CONT else 0,sm if c in CONT else 0,al)
        L[b]=s
    return L+np.r_[0,p['b1'],p['b2'],p['b3']]
def objective(tr):
    p=dict(nb_cont=tr.suggest_categorical('nb_cont',[12,16,20,24,32,40,48]),sm=tr.suggest_categorical('sm',[0.0,0.5,0.7,1.0,1.5]),
           alpha=tr.suggest_categorical('alpha',[0.3,1.0,3.0]),w_disc=tr.suggest_float('w_disc',0.6,1.4),
           b1=tr.suggest_float('b1',-0.3,0.3),b2=tr.suggest_float('b2',-0.3,0.3),b3=tr.suggest_float('b3',-0.3,0.3),
           **{f'use_{c}':tr.suggest_categorical(f'use_{c}',[True,False]) for c in WEAK})
    return f1(scores(p))
st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42,multivariate=True))
st.enqueue_trial(dict(nb_cont=32,sm=0.7,alpha=1.0,w_disc=1.0,b1=0,b2=0,b3=0,**{f'use_{c}':True for c in WEAK}))
t0=time.time()
st.optimize(objective,n_trials=int(sys.argv[1]),timeout=int(sys.argv[2]),callbacks=[lambda s,t_:print(f'trial {t_.number} {t_.value:.4f} best {s.best_value:.4f} {time.time()-t0:.0f}s',flush=True) if t_.number%10==0 else None])
print('BEST',round(st.best_value,4),st.best_params,flush=True)
json.dump(st.best_params,open('best_nb.json','w'))
# honesty check: same params on DIFFERENT folds (seed 7) vs baseline params
import nb as N
N.folds[:]=list(StratifiedKFold(5,shuffle=True,random_state=7).split(t,y));cache.clear()
base=dict(nb_cont=32,sm=0.7,alpha=1.0,w_disc=1.0,b1=0,b2=0,b3=0,**{f'use_{c}':True for c in WEAK})
print('fresh folds  baseline',round(f1(scores(base)),4),'| tuned',round(f1(scores(st.best_params)),4))
