"""Experimental request-owned state pool for the live MRv2 Higgs sampler.

This initial integration mirrors four scalar fields to the CPU in one copy
per step because the existing logits mask and LM termination consume them.
Sampling/RAS and their random draw order remain unchanged.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _advance(CODES, SLOTS, ACTION, ND, REM, DONE, COUNT, RING, FRAME, META,
             Q: tl.constexpr, W: tl.constexpr, BOS: tl.constexpr,
             EOS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    q = tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row)
    action = tl.load(ACTION + row)
    seed = action == 1
    fresh = (action == 1) | (action == 3)
    nd = tl.load(ND + slot)
    rem = tl.load(REM + slot)
    done = tl.load(DONE + slot)
    count = tl.load(COUNT + slot)
    nd = tl.where(fresh, 0, nd)
    rem = tl.where(fresh, -1, rem)
    done = tl.where(fresh, 0, done)
    count = tl.where(fresh, 0, count)
    codes = tl.load(CODES + row * Q + q, q < Q, other=0)
    leading = nd + 1 < Q
    codes = tl.where(leading & (q >= nd + 1), BOS, codes)
    next_nd = nd + leading.to(tl.int32)
    last_eos = tl.max(tl.where((q < Q) & (codes == EOS), q, -1), 0)
    locked = tl.where(rem >= 0, Q - rem, last_eos + 1)
    codes = tl.where(q < locked, EOS, codes)
    next_rem = tl.where(rem >= 0, rem - 1,
                        tl.where(last_eos >= 0, Q - last_eos - 1, -1))
    finished = next_rem == 0
    next_nd = tl.where(seed | finished, 0, next_nd)
    next_rem = tl.where(seed | finished, -1, next_rem)
    next_done = tl.where(seed, 0, done | finished.to(tl.int32))
    codes = tl.where(seed, BOS, codes)
    index = count % W
    # Duplicate the ring so the last <=7 frames are a chronological view.
    # Feedback reads the last column; RAS reads all codebooks' last 7 frames.
    ring_base = RING + slot * Q * (2 * W) + q * (2 * W)
    tl.store(ring_base + index, codes, q < Q)
    tl.store(ring_base + index + W, codes, q < Q)
    tl.store(FRAME + row * Q + q, codes, q < Q)
    tl.store(ND + slot, next_nd)
    tl.store(REM + slot, next_rem)
    tl.store(DONE + slot, next_done)
    tl.store(COUNT + slot, count + 1)
    tl.store(META + row * 4 + 0, next_nd)
    tl.store(META + row * 4 + 1, next_rem)
    tl.store(META + row * 4 + 2, next_done)
    tl.store(META + row * 4 + 3, count + 1)


class RequestStatePool:
    def __init__(self, device, q, *, capacity=256, window=7):
        self.q, self.window = q, window
        self.slots = {}
        self.free = list(reversed(range(capacity)))
        self.nd = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.rem = torch.full_like(self.nd, -1)
        self.done = torch.zeros_like(self.nd)
        self.count = torch.zeros_like(self.nd)
        self.ring = torch.empty((capacity, q, 2 * window), dtype=torch.long, device=device)
        self.calls = 0
        self.releases = 0

    def release(self, keys):
        for key in keys:
            slot = self.slots.pop(str(key), None)
            if slot is not None:
                self.free.append(slot)
                self.releases += 1

    def step(self, codes, audio_rows, seeded_rows, request_ids, states, *, bos, eos):
        rows = audio_rows + seeded_rows
        keys = [str(request_ids[row]) for row in rows]
        if len(set(keys)) != len(keys):
            raise RuntimeError('Duplicate live request ids in fused state step')
        actions, slots = [], []
        for key, row in zip(keys, rows, strict=True):
            fresh = key not in self.slots
            if fresh:
                if not self.free:
                    raise RuntimeError('Experimental fused state pool is full')
                self.slots[key] = self.free.pop()
            slots.append(self.slots[key])
            actions.append(1 if row in seeded_rows else (3 if fresh else 2))
        if not rows:
            return codes
        if seeded_rows:
            codes = torch.cat((codes, codes.new_zeros((len(seeded_rows), self.q))))
        slot_ids = torch.tensor(slots, dtype=torch.long, device=codes.device)
        action_ids = torch.tensor(actions, dtype=torch.int32, device=codes.device)
        frames = torch.empty_like(codes)
        meta = torch.empty((len(rows), 4), dtype=torch.int32, device=codes.device)
        _advance[(len(rows),)](codes, slot_ids, action_ids, self.nd, self.rem,
                              self.done, self.count, self.ring, frames, meta,
                              self.q, self.window, bos, eos,
                              triton.next_power_of_2(self.q), num_warps=1)
        # One batched D2H boundary. No eager replay is allowed after mutation.
        mirrors = meta.cpu().tolist()
        for key, slot, (nd, rem, done, count) in zip(keys, slots, mirrors, strict=True):
            start, stop = (0, count) if count < self.window else (
                count % self.window, count % self.window + self.window)
            state = states.setdefault(key, {})
            state.update(num_delay=nd, num_remaining_delays=None if rem < 0 else rem,
                         should_terminate=bool(done),
                         audio_out_ids=self.ring[slot, :, start:stop])
        self.calls += 1
        return frames[:len(audio_rows)]
