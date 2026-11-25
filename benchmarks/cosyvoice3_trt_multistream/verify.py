def _cv3_checked(self,items):
    import json
    from pathlib import Path
    count=getattr(self,"_cv3_check_n",0)
    if count>=16:return _cv3_dispatch(self,items)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        refs=[self.forward_streaming(token=item['token'],prompt_token=item['prompt_token'],prompt_feat=item['prompt_feat'],embedding=item['embedding'],cache_state=item.get('cache_state'),token_offset_tokens=int(item.get('token_offset_tokens',0)),n_timesteps=10,finalize=bool(item.get('finalize',False))) for item in items]
        ref_rng=torch.cuda.get_rng_state()
    obs=_cv3_dispatch(self,items)
    equal_rng=torch.equal(ref_rng,torch.cuda.get_rng_state())
    report=[]
    for (ra,rs),(a,s) in zip(refs,obs):
        rd=(ra.float()-a.float())
        state_eq=(s is None and rs is None) or (s is not None and rs is not None and set(s)==set(rs) and all(torch.equal(s[k],rs[k]) if torch.is_tensor(s[k]) else s[k]==rs[k] for k in s))
        report.append(dict(exact=bool(torch.equal(ra,a)),state_equal=bool(state_eq),rng_equal=bool(equal_rng),relative_l2=float((rd.norm()/ra.float().norm().clamp_min(1e-12)).item()),samples=a.numel()))
    path=Path(__import__('os').environ['CV3_VERIFY_FILE'])
    with path.open('a') as f:f.write(json.dumps(report)+'\n')
    self._cv3_check_n=count+len(report)
    return obs
CosyVoice3Code2Wav._cv3_dispatch=_cv3_checked
