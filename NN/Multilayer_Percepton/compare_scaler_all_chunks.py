"""Fit reusable Primary/Secondary scaler on ALL TRAIN rows, with resumable moments."""
from pathlib import Path
import json
import os
import time
import fcntl
import numpy as np
from dataset import (TRAIN, PRIMARY_DIM, SECONDARY_DIM, ModalityStandardizer,
                     _RunningMoments, discover_datasets, list_chunk_ids,
                     load_chunk_metadata, load_rows)
def summary(x):
    return {key: float(value) for key,value in
            zip(['median','p95','max'],np.percentile(np.asarray(x),[50,95,100]))}

BASE = Path(__file__).resolve().parent
OUT = BASE / 'analysis_scaler_2_vs_all'

def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'analysis.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    datasets = discover_datasets(BASE.parent.parent/'data_input_nn')
    jobs = [(i,int(c)) for i,d in enumerate(datasets) for c in list_chunk_ids(d)]
    manifest = [{'dataset':str(datasets[i]['name']),'chunk':c} for i,c in jobs]
    manifest_path = OUT/'chunks.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Chunk manifest changed; use a new output directory.')
    else:
        manifest_path.write_text(json.dumps(manifest,indent=2))
    p,s = _RunningMoments(PRIMARY_DIM),_RunningMoments(SECONDARY_DIM)
    state=OUT/'progress.npz'; offset=0
    if state.exists():
        with np.load(state) as data:
            offset=int(data['offset'])
            for prefix,stats in [('primary',p),('secondary',s)]:
                stats.count=int(data[prefix+'_count'])
                stats.mean[:]=data[prefix+'_mean']; stats.m2[:]=data[prefix+'_m2']
    print(f'ALL TRAIN ONLY | chunks={len(jobs)} | resume offset={offset} | rows={p.count}',flush=True)
    stores={}; start=time.time(); initial_offset=offset
    for index in range(offset,len(jobs)):
        i,c=jobs[index]; d=datasets[i]; meta=load_chunk_metadata(d,c,mmap=True)
        rows=np.flatnonzero(meta['split_id']==TRAIN)
        for k in range(0,len(rows),8192):
            r=rows[k:k+8192]
            x=load_rows(dataset=d,chunk_id=c,row_indices=r,group_ids=np.asarray(meta['group_id'][r],dtype=np.int64),feature_mode='primary_secondary',secondary_stores=stores,secondary_cache_size=2)
            p.update(x[:,:PRIMARY_DIM]); s.update(x[:,PRIMARY_DIM:])
        tmp=OUT/'progress.tmp.npz'
        np.savez(tmp,offset=index+1,primary_count=p.count,primary_mean=p.mean,primary_m2=p.m2,secondary_count=s.count,secondary_mean=s.mean,secondary_m2=s.m2)
        os.replace(tmp,state)
        elapsed=time.time()-start
        eta=elapsed/(index+1-initial_offset)*(len(jobs)-index-1)
        progress={'completed_chunks':index+1,'total_chunks':len(jobs),'train_rows':p.count,'elapsed_seconds_this_session':elapsed,'estimated_remaining_hours':eta/3600}
        (OUT/'status.tmp.json').write_text(json.dumps(progress,indent=2)); os.replace(OUT/'status.tmp.json',OUT/'status.json')
        print(f"{index+1}/{len(jobs)} | {d['name']} chunk={c} | total_rows={p.count:,} | ETA={eta/3600:.2f} h",flush=True)
    scaler=ModalityStandardizer(*p.finalize(),*s.finalize(),np.asarray([str(datasets[i]['name']) for i,c in jobs]),np.asarray([c for i,c in jobs]))
    scaler.save(OUT/'standardizer_all_train_chunks.npz')
    old=ModalityStandardizer.load(OUT/'standardizer_reference_2chunks.npz')
    result={'train_rows':p.count,'physical_chunks':len(jobs),'fit_split':'TRAIN only','statistics':{}}
    for name,mu2,sd2,muall,sdall in [('primary',old.primary_mean,old.primary_scale,scaler.primary_mean,scaler.primary_scale),('secondary',old.secondary_mean,old.secondary_scale,scaler.secondary_mean,scaler.secondary_scale)]:
        mean_shift=(muall.astype(np.float64)-mu2)/sdall
        relative_scale=np.abs(sd2.astype(np.float64)/sdall-1)
        # Exact aggregate E[(z_2-z_all)^2] over the fitted TRAIN population.
        scale_ratio=sdall.astype(np.float64)/sd2
        shift_in_old_units=(muall.astype(np.float64)-mu2)/sd2
        result['statistics'][name]={'mean_shift_in_sigma_all':summary(np.abs(mean_shift)),'relative_scale_difference':summary(relative_scale),'fraction_features_scale_diff_gt_10pct':float(np.mean(relative_scale>.1)),'rms_difference_z_on_all_train':float(np.sqrt(np.mean((scale_ratio-1)**2+shift_in_old_units**2)))}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
