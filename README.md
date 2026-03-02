# ternary-signal

The fundamental unit of communication in Blackfall neuromorphic systems.

```
s = p × m × k
```

Where `p` is polarity (`{-1, 0, +1}`), `m` is magnitude (`0–255`), and `k` is multiplier (`0–255`).

## Two Representations

### `Signal` — 3 bytes, full precision

```rust
use ternary_signal::Signal;

let excited = Signal::positive(200);            // pol=+1, mag=200, mul=1
let burst   = Signal::positive_amplified(200, 100); // pol=+1, mag=200, mul=100
let current = burst.current();                  // 20,000 (i32)
```

`Signal { polarity: i8, magnitude: u8, multiplier: u8 }` — `#[repr(C)]`, 3 bytes. Full 0–255 range on both magnitude and multiplier. Use this where you need component-level access or arithmetic operations (add, scale, decay, step-toward).

### `PackedSignal` — 1 byte, log-quantized

```rust
use ternary_signal::PackedSignal;

let packed = PackedSignal::pack(1, 200, 100);   // quantized to 3-bit codes
let sv     = packed.shift_value();               // i16-safe integration value
let exact  = packed.current();                   // full p×m×k via LUT (i32)
```

`PackedSignal(u8)` — `[pol:2|mag:3|mul:3]`, 1 byte. Magnitude and multiplier quantized to 8 log-scale levels via a frozen lookup table:

```
Code  000   001   010   011   100   101   110   111
Value   0     1     4    16    32    64   128   255
```

67% memory reduction over `Signal`. Designed for storage (synapses), transit (tracts, network), and integration (membrane accumulation).

## Shift-Weighted Integration

During membrane integration, `PackedSignal` avoids the full `p × m × k` multiplication. The multiplier's 3-bit code maps to a bit-shift:

```
Low mul  → right-shift (attenuate)
Mid mul  → no shift    (neutral)
High mul → left-shift  (amplify ×2, ×4, ×8)
```

Maximum single-signal contribution: `255 << 3 = 2040`. Stays within i16 range — no i32 accumulator needed.

```rust
use ternary_signal::PackedSignal;

// Membrane integration loop
let mut membrane: i16 = 0;
let signals: &[PackedSignal] = &[ /* incoming synaptic signals */ ];
for s in signals {
    membrane = membrane.saturating_add(s.shift_value());
}
```

## Conversion

Bidirectional, lossy in the pack direction (quantization):

```rust
use ternary_signal::{Signal, PackedSignal};

let signal = Signal::positive_amplified(200, 100);
let packed = PackedSignal::from(signal);       // lossy quantization
let back   = Signal::from(packed);             // exact from LUT values
```

## Signal Lifecycle

```
[1B packed] stored on synapse
[1B packed] transmitted across tract
[1B packed] arrives at dendrite
             ↓
             shift_value() → i16 into membrane accumulator (transient, per-neuron)
             ↓
             neuron fires → new PackedSignal on axon
```

## Properties

- `#![no_std]` — no allocator required
- All core methods are `const fn`
- Optional `serde` serialization (enabled by default)
- `LOG_LUT` is public and frozen — shared across the ecosystem

## Key Types

| Type | Size | Purpose |
|------|------|---------|
| `Polarity` | 1 byte | Enum: `{Negative, Zero, Positive}` |
| `Signal` | 3 bytes | Full-precision signal with component access |
| `PackedSignal` | 1 byte | Compact storage/transit format with LUT decode |
| `LOG_LUT` | 8 bytes | Frozen log-scale table: 3-bit code → `[0,1,4,16,32,64,128,255]` |

## License

MIT OR Apache-2.0
