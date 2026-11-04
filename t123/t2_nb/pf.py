import sys,time,json;sys.path.insert(0,'.');from nb import *
import optuna;optuna.logging.set_verbosity(optuna.logging.WARNING)
NB=[8,12,16,24,32,48,64];SM=[0,0.5,0.7,1.0,1.5,2.0];AL=[0.3,1.0,3.0]
FS={s:list(StratifiedKFold(5,shuffle=True,random_state=s).split(t,y)) for s in (42,7,123)}
cache={}
def LL(c,s,fo,nb_,sm,al):
    k=(c,s,fo,nb_,sm,al)
    if k not in cache:
        a,b=FS[s][fo]
        if len(cache)>2500:cache.clear()
        cache[k]=loglik(c,a,b,nbins=nb_,sm=sm,alpha=al).astype(np.float32)
    return cache[k]
def scores(cfg,al,s):
    L=np.zeros((len(y),K))
    for fo,(a,b) in enumerate(FS[s]):
        sc=np.log(np.bincount(y[a],minlength=K)/len(a))+np.zeros((len(b),K))
        for c in FEATS:
            if c in DISC: sc=sc+LL(c,s,fo,0,0,al)
            else: sc=sc+LL(c,s,fo,*cfg[c],al)
        L[b]=sc
    return L
def ev(cfg,al):return [round(f1(scores(cfg,al,s)),4) for s in (42,7,123)]
base={c:(32,0.7) for c in CONT}
def obj(tr):
    cfg={c:(tr.suggest_categorical(f'nb_{c}',NB),tr.suggest_categorical(f'sm_{c}',SM)) for c in CONT}
    return f1(scores(cfg,tr.suggest_categorical('alpha',AL),42))
st=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
st.enqueue_trial({**{f'nb_{c}':32 for c in CONT},**{f'sm_{c}':0.7 for c in CONT},'alpha':1.0})
st.optimize(obj,timeout=int(sys.argv[1]))
n=len(st.trials);bp=st.best_params
cfg={c:(bp[f'nb_{c}'],bp[f'sm_{c}']) for c in CONT};al=bp['alpha']
cache.clear();print('trials',n,'best',st.best_value);res={'baseline':ev(base,1.0),'optuna':ev(cfg,al)};print(res,flush=True)
# LL-criterion coordinate selection (alpha=1), seed-42 folds
cfg2={}
for c in CONT:
    best=None
    for nb_ in NB:
        for sm in SM:
            m=np.mean(np.concatenate([LL(c,42,fo,nb_,sm,1.0)[np.arange(len(b)),y[b]] for fo,(a,b) in enumerate(FS[42])]))
            if best is None or m>best[0]:best=(m,nb_,sm)
    cfg2[c]=(best[1],best[2])
res['ll_select']=ev(cfg2,1.0);print(res,cfg2)
json.dump({'trials':n,'results_f1_[seed42,seed7,seed123]':res,'optuna':{'alpha':al,'cfg':cfg},'ll_select':{'alpha':1.0,'cfg':cfg2}},open('best_nb_perfeature.json','w'),indent=1)
