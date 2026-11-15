exec(open('chk.py').read().split("# weak evidence")[0])
out=~U
nz=lambda m:np.nan_to_num(lr(m))
votes=np.c_[nz('accuracy_mean')>0.2,nz('kda_mean')>0.3,nz('reaction_time_mean')<-0.2,nz('skill_rating')>0.15,
            (X('skill_trajectory',4)-b123('skill_trajectory'))>12, (X('friend_removes',4)-b123('friend_removes'))>3,(X('friend_adds',4)-b123('friend_adds'))>6]
nv=votes.sum(1)
print('C votes dist inside C',np.bincount(nv[M[:,2]],minlength=8),' outside',np.bincount(nv[out],minlength=8))
sel=out&(nv>=3); print('outside rows with >=3 C votes',sel.sum(), 'votes',nv[sel])
lrloc=nz('location_changes');lrpf=nz('purchase_frequency');lrpam=nz('purchase_amount_mean')
aw=out&vpn&((lrpf>.5)|(lrpam>.35)|(lrloc>.5)); print('A weak',aw.sum(), np.c_[lrpf[aw],lrpam[aw],lrloc[aw],j3[aw],j4[aw]].round(2).tolist())
