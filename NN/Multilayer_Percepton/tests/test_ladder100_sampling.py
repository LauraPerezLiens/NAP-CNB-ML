"""Regression checks for bounded-memory monitoring (CPU, synthetic data)."""
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import mlp_layernorm_shuffled_ladder100_dualeval as m

class LowMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        m.DiskBinaryMetrics.directory=self.temp.name
        torch.set_num_threads(2)
    def tearDown(self):self.temp.cleanup()
    def test_exact_external_auc(self):
        rng=np.random.default_rng(12)
        for scores in [rng.normal(size=40000).astype('float32'),
                       rng.integers(-3,4,size=40000).astype('float32'),
                       np.zeros(40000,dtype='float32')]:
            labels=rng.integers(0,2,size=len(scores)).astype('float32')
            metric=m.DiskBinaryMetrics();metric.sort_memory_bytes=4096
            for start in range(0,len(labels),1024):
                metric.update(torch.from_numpy(scores[start:start+1024]),
                              torch.from_numpy(labels[start:start+1024]),1.)
            self.assertFalse(metric.y_true);self.assertFalse(metric.logits)
            self.assertAlmostEqual(metric.roc_auc(),roc_auc_score(labels,scores),places=13)
            metric.stream.flush()
            self.assertEqual(Path(metric.db_path).stat().st_size,len(labels)*5)
            metric.close(remove=True)
    def test_recovery_truncates_uncheckpointed_predictions(self):
        metric=m.DiskBinaryMetrics()
        metric.update(torch.tensor([-1.,1.,-0.,0.]),torch.tensor([0.,1.,1.,0.]),1.)
        state=metric.state_dict()
        metric.update(torch.tensor([3.]),torch.tensor([0.]),1.)
        metric.close()
        restored=m.DiskBinaryMetrics();restored.load_state_dict(state)
        self.assertEqual(restored.n,4)
        self.assertAlmostEqual(restored.roc_auc(),.875)
        restored.close(remove=True)
    def fixture(self):
        jobs=[m.ChunkJob(0,i) for i in range(3)]
        ys=[np.zeros(10,dtype=int),np.array([1]*4+[0]*6),np.array([0]*8+[1]*2)]
        def meta(d,c,mmap=True):return {'split_id':np.zeros(10,dtype=int),'y':ys[c]}
        def load(d,c,metadata,rows,*a):
            return np.column_stack([rows,rows]).astype('float32'),ys[c][rows].astype('float32')
        scaler=self.root/'scaler';scaler.write_bytes(b'fixture')
        args=SimpleNamespace(seed=42,output_dir=str(self.root),features='primary_secondary',
                             secondary_cache_size=1,eval_batch_size=3,auc_bins=4096,scaler_path=str(scaler),eval_target_per_class=3)
        return jobs,ys,meta,load,args
    def test_fixed_monitor_and_cache_reuse(self):
        jobs,ys,meta,load,args=self.fixture()
        with patch.object(m,'load_chunk_metadata',side_effect=meta),patch.object(m,'_load_chunk_rows',side_effect=load),patch.object(m,'transform_input',side_effect=lambda X,*a:X):
            cache=m.build_monitor_cache([{'name':'fixture'}],jobs,0,args,None)
            rows=json.loads((self.root/'monitor_rows_split_0.json').read_text())
            self.assertTrue(all(r['rows'] and len(set(r['rows']))==len(r['rows']) for r in rows))
            model=m.MLP(2);model.train();rng=torch.get_rng_state().clone()
            first=m.evaluate_monitor(model,cache,args,torch.device('cpu'))
            second=m.evaluate_monitor(model,cache,args,torch.device('cpu'))
            np.testing.assert_allclose(list(first.values()),list(second.values()),equal_nan=True)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(first['tp']+first['fn'],first['tn']+first['fp'])
            with patch.object(m,'load_chunk_metadata',side_effect=AssertionError('Cache should be reused')):
                self.assertEqual(cache,m.build_monitor_cache([{'name':'fixture'}],jobs,0,args,None))
    def test_disk_plan_balanced_unique_and_reused(self):
        jobs,ys,meta,load,args=self.fixture()
        with patch.object(m,'load_chunk_metadata',side_effect=meta):
            plan=m.build_disk_eval_plan([{'name':'fixture'}],jobs,42,0,self.root/'plan')
            labels=np.concatenate([ys[j.chunk_id][plan[j]] for j in jobs])
            self.assertEqual(int((labels==0).sum()),6)
            self.assertEqual(int((labels==1).sum()),6)
            for job in jobs:self.assertEqual(len(set(plan[job])),len(plan[job]))
        with patch.object(m,'load_chunk_metadata',side_effect=AssertionError('Plan should be reused')):
            m.build_disk_eval_plan([{'name':'fixture'}],jobs,42,0,self.root/'plan')
    def test_atomic_checkpoint_contains_rng(self):
        args=SimpleNamespace(features='primary_secondary',dropout=.3,seed=42)
        model=m.MLP(1792);optimizer=torch.optim.AdamW(model.parameters())
        scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,20)
        path=self.root/'checkpoint.pt'
        m.save_checkpoint(path,model,optimizer,scheduler,args,0,1,3,0)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        self.assertIn('torch_rng_state',checkpoint)
        self.assertFalse(Path(str(path)+'.tmp').exists())

if __name__=='__main__':unittest.main()
