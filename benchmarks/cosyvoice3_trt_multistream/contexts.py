from pathlib import Path
import json, time, statistics
import torch,tensorrt as trt
p=Path(__import__('os').environ['CV3_EXPERIMENT_DIR'])
logger=trt.Logger(trt.Logger.ERROR)
runtime=trt.Runtime(logger)
engine=runtime.deserialize_cuda_engine(next((p/'plans').glob('flow_estimator_*.plan')).read_bytes())
assert engine
# Each logical request has its own immutable input tensors and private output.
names=[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
outname=next(n for n in names if engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT)
ctx=[engine.create_execution_context() for _ in range(4)]
assert all(ctx)
streams=[torch.cuda.Stream() for _ in ctx]
torch.manual_seed(42)
result={}
for widths in [(450,450,450,450),(450,510,630,750)]:
    data=[]
    for width in widths:
        shape=dict(x=(2,80,width),mask=(2,1,width),mu=(2,80,width),t=(2,),spks=(2,80),cond=(2,80,width))
        inp={n:torch.randn(dim,device='cuda') for n,dim in shape.items()}
        inp['mask'].fill_(1);inp['t'].fill_(0.5)
        inp[outname]=torch.empty_like(inp['x'])
        data.append(inp)
    torch.cuda.synchronize()
    def run(count):
        # Four private contexts even in the serial control: stream count and
        # submission order are the only differences. Bindings stay fixed.
        for i,c in enumerate(ctx):
            for n,v in data[i].items():
                if n!=outname:assert c.set_input_shape(n,tuple(v.shape))
                assert c.set_tensor_address(n,v.data_ptr())
        t0=time.perf_counter()
        for _ in range(10):
            for i,c in enumerate(ctx):
                assert c.execute_async_v3(streams[i%count].cuda_stream)
        torch.cuda.synchronize()
        return (time.perf_counter()-t0)*1000
    for k in (1,2,4):run(k)
    ref=[v[outname].clone() for v in data]
    torch.cuda.synchronize()
    timings={k:[] for k in (1,2,4)}
    for k in (1,2,4,4,2,1)*4:
        timings[k].append(run(k))
    exact=all(torch.equal(v[outname],r) and bool(r.isfinite().all()) for v,r in zip(data,ref))
    result[str(widths)]=dict(ms=timings,median={k:statistics.median(x) for k,x in timings.items()},finite_and_exact=exact)
    print(result[str(widths)]['median'],exact,flush=True)
(p/'contexts.json').write_text(json.dumps(result,indent=2))
