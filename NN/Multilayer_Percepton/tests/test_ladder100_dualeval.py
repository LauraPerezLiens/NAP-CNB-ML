"""Verify dual evaluation and selection by local TEST, including intra-epoch resume."""
import contextlib,io,json,sys,tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import torch
import mlp_layernorm_shuffled_ladder100_dualeval as m

class ReachedEpochTwo(Exception):pass

def run_test():
 with tempfile.TemporaryDirectory() as out:
  with patch.object(sys,'argv',['test','--output-dir',out,'--epochs','2','--max-chunks','100','--report-every-chunks','1']):args=m.parse_args()
  torch.set_num_threads(2)
  model=m.MLP(1792,args.dropout)
  opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
  scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,args.scheduler_t0,T_mult=args.scheduler_tmult,eta_min=args.scheduler_min_lr)
  metrics=m.StreamingBinaryMetrics()
  metrics.update(torch.tensor([-1.,1.,-2.,2.]),torch.tensor([0.,1.,0.,1.]),1.)
  state=m.EpochScalarMetrics();state.merge(metrics)
  recovery=Path(out)/'recovery_mlp_primary_secondary.pt'
  m.save_checkpoint(recovery,model,opt,scheduler,args,0,97,100,0,train_epoch_metrics_state=state.state_dict())
  args.resume=str(recovery)
  all_jobs=[m.ChunkJob(0,i) for i in range(200)]
  trained=[];sampled=[];evaluated=[]
  def cache(datasets,jobs,split,sample_args,standardizer):
   path=Path(sample_args.output_dir)/f'monitor_batches_split_{split}.pkl'
   path.touch();path.with_suffix('.ready.json').write_text(json.dumps({'n':4}))
   sampled.append((len(jobs),split));return path
  def evaluate(model,path,args,device):
   values=metrics.compute().copy()
   values['loss']=.9 if path.parent.name=='shared_global_eval' else .25
   evaluated.append((path.parent.name,values['loss']));return values
  def train(**kw):
   trained.append((kw['epoch'],kw['block_jobs'][0]))
   if kw['epoch']==1:raise ReachedEpochTwo()
   return metrics
  output=io.StringIO()
  with patch.object(m,'parse_args',return_value=args),patch.object(m,'distributed_setup',return_value=(False,0,0,1,torch.device('cpu'))),patch.object(m,'discover_datasets',return_value=[{'name':'fixture'}]),patch.object(m,'build_chunk_jobs',return_value=all_jobs),patch.object(m,'prepare_standardizer',return_value=None),patch.object(m,'build_monitor_cache',side_effect=cache),patch.object(m,'train_block',side_effect=train),patch.object(m,'evaluate_monitor',side_effect=evaluate),patch.object(m,'evaluate_split',side_effect=AssertionError('Full evaluation forbidden')),patch.object(m,'build_disk_eval_plan',side_effect=AssertionError('No full-data plans')),contextlib.redirect_stdout(output):
   try:m.main()
   except ReachedEpochTwo:pass
   else:raise AssertionError('Epoch 2 not reached')
  expected=m.ordered_epoch_jobs(m.select_fixed_jobs(all_jobs,42,100,111111),42,0)[-3:]
  assert [j for e,j in trained[:3]]==expected
  assert len(trained)==4 and trained[-1][0]==1
  assert sampled==[(100,0),(100,2),(200,0),(200,2)],sampled
  assert len(evaluated)==12
  for name in ['best','last']:
   c=torch.load(Path(out)/f'{name}_mlp_primary_secondary.pt',map_location='cpu',weights_only=False)
   assert c['completed_epochs']==1 and c['best_val_loss']==.25
   assert c['selection_metric']=='stage_fixed_50_50_test_loss'
  history=json.loads((Path(out)/'history_mlp_primary_secondary.jsonl').read_text())
  assert history['test_eval_stage']['loss']==.25 and history['test_eval_all']['loss']==.9
  assert history['stage_chunks']==100 and history['all_chunks']==200
  assert 'positive=2 | negative=2' in output.getvalue()
  assert 'EPOCH 2/2' in output.getvalue()
 print('PASS: four fixed eval sets; balanced counts in log; best uses stage loss, not global loss; recovery and epoch 2 work.')

if __name__=='__main__':run_test()
