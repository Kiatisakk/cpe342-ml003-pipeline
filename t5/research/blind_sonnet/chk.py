import warnings;warnings.filterwarnings('ignore')
exec(open('rules.py').read().split("G={")[0])
M=np.load('out/rule_M.npy');U=M.any(1)
vpn=d.vpn_usage.values==1; pw=d.password_changed.values==1; ms=d.mass_item_sale.values==1
print('A: vpn1',vpn[M[:,0]].sum(),'/',M[:,0].sum(),' NaN vpn',d.vpn_usage[M[:,0]].isna().sum())
print('B: pw1',pw[M[:,1]].sum(),'/',M[:,1].sum(),' NaN pw',d.password_changed[M[:,1]].isna().sum())
# weak evidence outside groups
lrloc=np.nan_to_num(lr('location_changes')); lrcur=np.nan_to_num(lr('currency_transfers'))
lrses=np.nan_to_num(lr('session_duration_mean')); lrplay=np.nan_to_num(lr('play_pattern_consistency'))
lrpf=np.nan_to_num(lr('purchase_frequency')); lrpam=np.nan_to_num(lr('purchase_amount_mean'))
for nm,flag in (('vpn',vpn),('pw',pw)):
    print(nm,'rows total',flag.sum(),'outside union',(flag&~U).sum())
out=~U
print('outside union & vpn: with geo3/4 jump',(out&vpn&(g3|g4)).sum(),' pf>.5',(out&vpn&(lrpf>.5)).sum(),' pam>.35',(out&vpn&(lrpam>.35)).sum(),'loc>.5',(out&vpn&(lrloc>.5)).sum(),'j3>3',(out&vpn&(j3>3)).sum())
print('outside union & pw: geo4',(out&pw&g4).sum(),'play<-.2',(out&pw&(lrplay<-.2)).sum(),'loc>.5',(out&pw&(lrloc>.5)).sum(),'ses<-.4',(out&pw&(lrses<-.4)).sum(),'cur>.7',(out&pw&(lrcur>.7)).sum())
print('outside union & ms: ',(out&ms&(g4|(lrplay<-.2))).sum())
# B rows without pw: check
print('B rows pw!=1:',d.password_changed[M[:,1]].value_counts(dropna=False).to_dict(), ' ms:',d.mass_item_sale[M[:,1]].value_counts(dropna=False).to_dict())
print('A rows vpn:',d.vpn_usage[M[:,0]].value_counts(dropna=False).to_dict())
# B composition
print('B: play',(play&M[:,1]).sum(),'g4',(g4&M[:,1]).sum(),'g4only',(M[:,1]&~play).sum())
print('B rows play-only (no g4)',(M[:,1]&play&~g4).sum(), ' A rows g3',(M[:,0]&g3).sum(),'A by pf&pam only',(M[:,0]&~g3).sum())
