def _cv3_dispatch(self, items):
    device=next(self.flow_model.parameters()).device
    if not hasattr(self,"_cv3_lanes"):
        self._cv3_lanes=[torch.cuda.Stream(device=device) for _ in range(4)]
    caller=torch.cuda.current_stream(device)
    done=[];results=[]
    self._cv3_overlap=True
    try:
        for i,item in enumerate(items):
            lane=self._cv3_lanes[i%len(self._cv3_lanes)]
            lane.wait_stream(caller)
            with torch.cuda.stream(lane):
                for tensor in (item['token'],item['prompt_token'],item['prompt_feat'],item['embedding']):
                    if tensor.is_cuda:tensor.record_stream(lane)
                print(f"CV3BEGIN {item['req_id']} {item['token_offset_tokens']} {item['token'].shape[-1]} {__import__('time').monotonic_ns()}",flush=True)
                audio,state=self.forward_streaming(token=item['token'],prompt_token=item['prompt_token'],prompt_feat=item['prompt_feat'],embedding=item['embedding'],cache_state=item.get('cache_state'),token_offset_tokens=int(item.get('token_offset_tokens',0)),n_timesteps=10,finalize=bool(item.get('finalize',False)))
                results.append((audio,state))
                done.append(lane.record_event())
    finally:
        self._cv3_overlap=False
    for (audio,state),event in zip(results,done):
        caller.wait_event(event);audio.record_stream(caller)
        if state:
            for key in ('mel','phase_acc'):
                x=state.get(key)
                if isinstance(x,torch.Tensor) and x.is_cuda:
                    x.record_stream(caller)
                    state[key]=x.detach().cpu().contiguous()
    return results
CosyVoice3Code2Wav._cv3_dispatch=_cv3_dispatch
