//! # ternary-signal
//!
//! The fundamental unit of ternary neural communication: `Signal = polarity × magnitude × multiplier`
//!
//! Compact 3-byte representation combining:
//! - **Polarity**: excitatory (+1), inhibitory (-1), or silent (0)
//! - **Magnitude**: base intensity 0–255 (intrinsic neuron drive)
//! - **Multiplier**: contextual scaling 0–255 (metabolic state, arousal, burst energy)
//!
//! Effective magnitude = `magnitude × multiplier` (0–65,025 range).
//!
//! This is the canonical Signal type shared across neuromorphic crates
//! (ternsig, thermogram, astromind-snn, etc.).
//!
//! # Example
//! ```
//! use ternary_signal::{Signal, Polarity};
//!
//! let excited = Signal::positive(200);
//! assert!(excited.is_positive());
//! assert_eq!(excited.effective_magnitude(), 200); // multiplier defaults to 1
//!
//! let burst = Signal::positive_amplified(200, 100);
//! assert_eq!(burst.effective_magnitude(), 20_000); // 200 × 100
//!
//! let inhibited = Signal::with_polarity(Polarity::Negative, 50);
//! assert!(inhibited.is_negative());
//!
//! let neutral = Signal::zero();
//! assert!(!neutral.is_active());
//! ```

#![no_std]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Polarity
// ---------------------------------------------------------------------------

/// Polarity of a neural signal — strictly {-1, 0, +1}
///
/// Using this enum instead of raw `i8` prevents invalid states like polarity=2.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(i8)]
pub enum Polarity {
    /// Inhibitory signal
    Negative = -1,
    /// No signal / silent
    #[default]
    Zero = 0,
    /// Excitatory signal
    Positive = 1,
}

impl Polarity {
    /// Convert to i8
    #[inline]
    pub const fn as_i8(self) -> i8 {
        self as i8
    }

    /// Try to convert from i8, returns None for invalid values
    #[inline]
    pub const fn from_i8(value: i8) -> Option<Self> {
        match value {
            -1 => Some(Self::Negative),
            0 => Some(Self::Zero),
            1 => Some(Self::Positive),
            _ => None,
        }
    }

    /// Convert from i8, clamping invalid values to the nearest valid polarity
    #[inline]
    pub const fn from_i8_clamped(value: i8) -> Self {
        if value > 0 {
            Self::Positive
        } else if value < 0 {
            Self::Negative
        } else {
            Self::Zero
        }
    }

    /// Is this an active (non-zero) polarity?
    #[inline]
    pub const fn is_active(self) -> bool {
        !matches!(self, Self::Zero)
    }
}

impl From<Polarity> for i8 {
    fn from(p: Polarity) -> i8 {
        p.as_i8()
    }
}

impl TryFrom<i8> for Polarity {
    type Error = &'static str;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        Polarity::from_i8(value).ok_or("polarity must be -1, 0, or +1")
    }
}

// ---------------------------------------------------------------------------
// Signal
// ---------------------------------------------------------------------------

/// Signal: polarity × magnitude × multiplier (`s = p × m × k`)
///
/// The fundamental unit of neural communication.
/// Compact 3-byte `#[repr(C)]` representation:
/// - `polarity`: -1 (inhibited), 0 (neutral), +1 (excited)
/// - `magnitude`: 0–255 (base intensity — intrinsic neuron drive)
/// - `multiplier`: 0–255 (contextual scaling — metabolic state, arousal, burst energy)
///
/// Effective magnitude = `magnitude × multiplier` (range 0–65,025).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Signal {
    /// Polarity: -1 (inhibited), 0 (neutral), +1 (excited)
    pub polarity: i8,
    /// Magnitude: 0–255 (base intensity — intrinsic neuron drive)
    pub magnitude: u8,
    /// Multiplier: 0–255 (contextual scaling — metabolic state, arousal, burst energy)
    ///
    /// Default: 1 (neutral — effective magnitude equals base magnitude).
    /// 0 = silenced (effective magnitude is zero regardless of base magnitude).
    pub multiplier: u8,
}

impl Signal {
    // -- Constants -----------------------------------------------------------

    /// Zero signal (no activity)
    pub const ZERO: Self = Self {
        polarity: 0,
        magnitude: 0,
        multiplier: 0,
    };
    /// Maximum positive signal
    pub const MAX_POSITIVE: Self = Self {
        polarity: 1,
        magnitude: 255,
        multiplier: 255,
    };
    /// Maximum negative signal
    pub const MAX_NEGATIVE: Self = Self {
        polarity: -1,
        magnitude: 255,
        multiplier: 255,
    };

    // -- Effective magnitude -------------------------------------------------

    /// Compute effective magnitude: `magnitude × multiplier` (0–65,025)
    #[inline]
    pub const fn effective_magnitude(&self) -> u16 {
        self.magnitude as u16 * self.multiplier as u16
    }

    /// Compute signed effective current: `polarity × magnitude × multiplier`
    #[inline]
    pub const fn current(&self) -> i32 {
        self.polarity as i32 * self.effective_magnitude() as i32
    }

    // -- Constructors --------------------------------------------------------

    /// Create a zero / neutral signal
    #[inline]
    pub const fn zero() -> Self {
        Self::ZERO
    }

    /// Create a positive (excitatory) signal with multiplier = 1
    #[inline]
    pub const fn positive(magnitude: u8) -> Self {
        Self {
            polarity: 1,
            magnitude,
            multiplier: 1,
        }
    }

    /// Create a negative (inhibitory) signal with multiplier = 1
    #[inline]
    pub const fn negative(magnitude: u8) -> Self {
        Self {
            polarity: -1,
            magnitude,
            multiplier: 1,
        }
    }

    /// Create a positive signal with explicit multiplier
    #[inline]
    pub const fn positive_amplified(magnitude: u8, multiplier: u8) -> Self {
        Self {
            polarity: 1,
            magnitude,
            multiplier,
        }
    }

    /// Create a negative signal with explicit multiplier
    #[inline]
    pub const fn negative_amplified(magnitude: u8, multiplier: u8) -> Self {
        Self {
            polarity: -1,
            magnitude,
            multiplier,
        }
    }

    /// Create from [`Polarity`] enum and magnitude (type-safe, multiplier = 1)
    #[inline]
    pub const fn with_polarity(polarity: Polarity, magnitude: u8) -> Self {
        Self {
            polarity: polarity.as_i8(),
            magnitude,
            multiplier: 1,
        }
    }

    /// Create from [`Polarity`] enum, magnitude, and multiplier (type-safe)
    #[inline]
    pub const fn with_polarity_amplified(
        polarity: Polarity,
        magnitude: u8,
        multiplier: u8,
    ) -> Self {
        Self {
            polarity: polarity.as_i8(),
            magnitude,
            multiplier,
        }
    }

    /// Create from raw `i8` polarity and magnitude (**unchecked**, multiplier = 1)
    #[inline]
    pub const fn new(polarity: i8, magnitude: u8) -> Self {
        Self {
            polarity,
            magnitude,
            multiplier: 1,
        }
    }

    /// Create from raw components (**unchecked**)
    #[inline]
    pub const fn new_raw(polarity: i8, magnitude: u8, multiplier: u8) -> Self {
        Self {
            polarity,
            magnitude,
            multiplier,
        }
    }

    /// Create from raw `i8` polarity with validation (multiplier = 1)
    #[inline]
    pub const fn new_checked(polarity: i8, magnitude: u8) -> Option<Self> {
        match Polarity::from_i8(polarity) {
            Some(_) => Some(Self {
                polarity,
                magnitude,
                multiplier: 1,
            }),
            None => None,
        }
    }

    /// Create from separate float components (multiplier = 1)
    ///
    /// `polarity_f`: quantised to {-1, 0, +1} (dead-zone ±0.1)
    /// `magnitude_f`: clamped to 0.0–1.0, scaled to 0–255
    #[inline]
    pub fn from_floats(polarity_f: f32, magnitude_f: f32) -> Self {
        let polarity = if polarity_f > 0.1 {
            1
        } else if polarity_f < -0.1 {
            -1
        } else {
            0
        };
        let magnitude = (clamp_f32(magnitude_f, 0.0, 1.0) * 255.0) as u8;
        Self {
            polarity,
            magnitude,
            multiplier: 1,
        }
    }

    /// Create from a single signed float (-1.0 to 1.0), multiplier = 1
    ///
    /// Sign → polarity, absolute value → magnitude.
    #[inline]
    pub fn from_signed(value: f32) -> Self {
        let polarity = if value > 0.01 {
            1
        } else if value < -0.01 {
            -1
        } else {
            0
        };
        let abs = if value < 0.0 { -value } else { value };
        let clamped = if abs > 1.0 { 1.0 } else { abs };
        let magnitude = (clamped * 255.0) as u8;
        Self {
            polarity,
            magnitude,
            multiplier: 1,
        }
    }

    /// Create from a signed `i32` (multiplier = 1, clamps magnitude to 255).
    ///
    /// **Warning**: This discards values above ±255. For the full ±65,025 range,
    /// use [`from_current`] instead.
    #[inline]
    pub fn from_signed_i32(value: i32) -> Self {
        if value == 0 {
            Self::ZERO
        } else if value > 0 {
            Self {
                polarity: 1,
                magnitude: (if value > 255 { 255 } else { value }) as u8,
                multiplier: 1,
            }
        } else {
            let abs = if value == i32::MIN { 255 } else { -value };
            Self {
                polarity: -1,
                magnitude: (if abs > 255 { 255 } else { abs }) as u8,
                multiplier: 1,
            }
        }
    }

    /// Create from a signed `i32` using the full `p × m × k` range (±65,025).
    ///
    /// Decomposes the absolute value into `magnitude × multiplier` where both
    /// are 0–255. Values above 65,025 are clamped. Values ≤ 255 get multiplier=1.
    /// Values > 255 use magnitude=255 and multiplier=ceil(abs/255).
    #[inline]
    pub fn from_current(value: i32) -> Self {
        if value == 0 {
            return Self::ZERO;
        }

        let polarity: i8 = if value > 0 { 1 } else { -1 };
        let abs = if value == i32::MIN {
            65025u32
        } else if value < 0 {
            (-value) as u32
        } else {
            value as u32
        };

        if abs <= 255 {
            Self {
                polarity,
                magnitude: abs as u8,
                multiplier: 1,
            }
        } else {
            // abs > 255: use magnitude=255, multiplier = abs/255 (rounded)
            let clamped = if abs > 65025 { 65025 } else { abs };
            let mul = ((clamped + 127) / 255).min(255) as u8;
            let mul = if mul == 0 { 1 } else { mul };
            // Recompute magnitude to minimize rounding error: mag = clamped / mul
            let mag = (clamped / mul as u32).min(255) as u8;
            Self {
                polarity,
                magnitude: mag,
                multiplier: mul,
            }
        }
    }

    /// Create from a signed `i16` (multiplier = 1)
    #[inline]
    pub fn from_i16(value: i16) -> Self {
        if value == 0 {
            Self::ZERO
        } else if value > 0 {
            Self {
                polarity: 1,
                magnitude: (if value > 255 { 255 } else { value }) as u8,
                multiplier: 1,
            }
        } else {
            let abs = -(value as i32);
            Self {
                polarity: -1,
                magnitude: (if abs > 255 { 255 } else { abs }) as u8,
                multiplier: 1,
            }
        }
    }

    /// Create from u8 bipolar representation (128 = baseline)
    #[inline]
    pub fn from_u8_bipolar(level: u8) -> Self {
        let delta = level as i16 - 128;
        Self::from_i16(delta)
    }

    /// Create from spike rate (Hz)
    #[inline]
    pub fn from_spike_rate(rate_hz: f32, max_rate_hz: f32) -> Self {
        if rate_hz <= 0.0 || max_rate_hz <= 0.0 {
            return Self::ZERO;
        }
        let normalized = clamp_f32(rate_hz / max_rate_hz, 0.0, 1.0);
        let magnitude = (normalized * 255.0) as u8;
        if magnitude == 0 {
            Self::ZERO
        } else {
            Self::positive(magnitude)
        }
    }

    /// Create from spike count in a time window
    #[inline]
    pub fn from_spike_count(count: u32, window_ms: f32, max_rate_hz: f32) -> Self {
        if count == 0 || window_ms <= 0.0 {
            return Self::ZERO;
        }
        let rate_hz = count as f32 * 1000.0 / window_ms;
        Self::from_spike_rate(rate_hz, max_rate_hz)
    }

    // -- Conversion ----------------------------------------------------------

    /// Get as signed `i32` using effective magnitude (`polarity × magnitude × multiplier`)
    #[inline]
    pub fn as_signed_i32(&self) -> i32 {
        self.current()
    }

    /// Get base magnitude as float (0.0–1.0)
    #[inline]
    pub fn magnitude_f32(&self) -> f32 {
        self.magnitude as f32 / 255.0
    }

    /// Get effective magnitude as float (0.0–1.0, normalized to max 65025)
    #[inline]
    pub fn effective_magnitude_f32(&self) -> f32 {
        self.effective_magnitude() as f32 / 65025.0
    }

    /// Get as signed float using effective magnitude (-1.0 to 1.0)
    #[inline]
    pub fn as_signed_f32(&self) -> f32 {
        self.polarity as f32 * self.effective_magnitude_f32()
    }

    /// Convert to u8 bipolar (128 = baseline), using base magnitude only
    #[inline]
    pub fn to_u8_bipolar(&self) -> u8 {
        let signed = self.polarity as i16 * self.magnitude as i16;
        let result = signed + 128;
        (if result < 0 {
            0
        } else if result > 255 {
            255
        } else {
            result
        }) as u8
    }

    /// Convert to spike rate (Hz), using base magnitude
    #[inline]
    pub fn to_spike_rate(&self, max_rate_hz: f32) -> f32 {
        if self.polarity == 0 || self.magnitude == 0 {
            return 0.0;
        }
        let normalized = self.magnitude as f32 / 255.0;
        normalized * max_rate_hz * self.polarity.signum() as f32
    }

    /// Get polarity as [`Polarity`] enum (type-safe)
    #[inline]
    pub fn get_polarity(&self) -> Polarity {
        Polarity::from_i8(self.polarity).unwrap_or(Polarity::Zero)
    }

    // -- Queries -------------------------------------------------------------

    /// Is this signal active (non-zero)?
    #[inline]
    pub fn is_active(&self) -> bool {
        self.polarity != 0 && self.magnitude > 0
    }

    /// Is this a positive / excitatory signal?
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.polarity > 0 && self.magnitude > 0
    }

    /// Is this a negative / inhibitory signal?
    #[inline]
    pub fn is_negative(&self) -> bool {
        self.polarity < 0 && self.magnitude > 0
    }

    // -- Multiplier operations -----------------------------------------------

    /// Set the multiplier (contextual scaling)
    #[inline]
    pub const fn with_multiplier(self, multiplier: u8) -> Self {
        Self {
            polarity: self.polarity,
            magnitude: self.magnitude,
            multiplier,
        }
    }

    /// Amplify: increase multiplier by delta (saturating)
    #[inline]
    pub const fn amplify(self, delta: u8) -> Self {
        Self {
            polarity: self.polarity,
            magnitude: self.magnitude,
            multiplier: self.multiplier.saturating_add(delta),
        }
    }

    /// Attenuate: decrease multiplier by delta (saturating)
    #[inline]
    pub const fn attenuate(self, delta: u8) -> Self {
        Self {
            polarity: self.polarity,
            magnitude: self.magnitude,
            multiplier: self.multiplier.saturating_sub(delta),
        }
    }

    // -- Arithmetic ----------------------------------------------------------

    /// Add two signals (same polarity adds, opposite cancels).
    /// Operates on base magnitude. Result gets multiplier = 1.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let a = self.polarity as i32 * self.magnitude as i32;
        let b = other.polarity as i32 * other.magnitude as i32;
        Self::from_signed_i32(a + b)
    }

    /// Scale base magnitude by a factor (multiplier unchanged)
    #[inline]
    pub fn scale(&self, factor: f32) -> Self {
        let new_mag = clamp_f32(self.magnitude as f32 * factor, 0.0, 255.0) as u8;
        Self {
            polarity: if new_mag > 0 { self.polarity } else { 0 },
            magnitude: new_mag,
            multiplier: self.multiplier,
        }
    }

    /// Apply decay to base magnitude (in-place, for temporal fields)
    #[inline]
    pub fn decay(&mut self, retention: f32) {
        self.magnitude = (self.magnitude as f32 * retention) as u8;
        if self.magnitude == 0 {
            self.polarity = 0;
        }
    }

    /// Return a decayed copy
    #[inline]
    pub fn decayed(&self, retention: f32) -> Self {
        let mut copy = *self;
        copy.decay(retention);
        copy
    }

    // -- Smooth transitions --------------------------------------------------

    /// Step toward target by one unit (base magnitude)
    #[inline]
    pub fn step_toward(&self, target: &Self) -> Self {
        self.step_toward_by(target, 1)
    }

    /// Step toward target by specified delta (base magnitude)
    #[inline]
    pub fn step_toward_by(&self, target: &Self, delta: u8) -> Self {
        let current = self.polarity as i16 * self.magnitude as i16;
        let target_val = target.polarity as i16 * target.magnitude as i16;
        let diff = target_val - current;

        if diff.unsigned_abs() <= delta as u16 {
            *target
        } else if diff > 0 {
            let mut result = Self::from_signed_i32((current + delta as i16) as i32);
            result.multiplier = self.multiplier;
            result
        } else {
            let mut result = Self::from_signed_i32((current - delta as i16) as i32);
            result.multiplier = self.multiplier;
            result
        }
    }

    /// Step toward target by fractional amount (0.0–1.0)
    #[inline]
    pub fn step_toward_ratio(&self, target: &Self, ratio: f32) -> Self {
        let current = self.polarity as i16 * self.magnitude as i16;
        let target_val = target.polarity as i16 * target.magnitude as i16;
        let diff = target_val - current;
        let abs_diff = if diff < 0 { -diff } else { diff } as f32;
        let clamped = clamp_f32(ratio, 0.0, 1.0);
        let delta = {
            let raw = (abs_diff * clamped + 0.999) as i16; // ceil
            if raw < 1 {
                1
            } else {
                raw
            }
        };
        self.step_toward_by(target, delta as u8)
    }

    /// Check if this signal has reached the target (within tolerance, base magnitude)
    #[inline]
    pub fn reached(&self, target: &Self, tolerance: u8) -> bool {
        let current = self.polarity as i16 * self.magnitude as i16;
        let target_val = target.polarity as i16 * target.magnitude as i16;
        let diff = current - target_val;
        let abs_diff = if diff < 0 { -diff } else { diff };
        abs_diff <= tolerance as i16
    }
}

// ---------------------------------------------------------------------------
// PackedSignal — 1-byte compact format
// ---------------------------------------------------------------------------

/// Log-scale lookup table: 3-bit code → effective value (0–255).
///
/// Shared by both magnitude and multiplier fields — same encoding, single table.
/// Logarithmic spacing preserves perceptual significance:
/// fine granularity at low intensities, coarser at high.
/// This table is frozen and will never change.
pub const LOG_LUT: [u8; 8] = [0, 1, 4, 16, 32, 64, 128, 255];

/// Shift table: 3-bit multiplier code → bit-shift for i16-safe integration.
///
/// Low multiplier attenuates (right-shift), high multiplier amplifies (left-shift).
/// Maximum single-signal contribution: `255 << 3 = 2040`, well within i16 range.
const SHIFT_TABLE: [i8; 8] = [
    0,  // code 0: mul=0, signal dead (magnitude also 0 for this code)
    -2, // code 1: mul=1, attenuate >>2
    -1, // code 2: mul=4, attenuate >>1
    0,  // code 3: mul=16, neutral
    0,  // code 4: mul=32, neutral
    1,  // code 5: mul=64, amplify <<1
    2,  // code 6: mul=128, amplify <<2
    3,  // code 7: mul=255, amplify <<3
];

/// Precomputed `polarity × magnitude × multiplier` for every possible packed byte.
///
/// 256 entries × 4 bytes = 1 KB — fits in L1 cache.
/// Use this for operations that need the true scalar current (similarity, TERNARY_MATMUL,
/// metabolic cost) outside the integration hot path.
const CURRENT_LUT: [i32; 256] = {
    let mut lut = [0i32; 256];
    let mut i = 0u16;
    while i < 256 {
        let byte = i as u8;
        let pol: i32 = match byte >> 6 {
            1 => 1,
            2 => -1,
            _ => 0,
        };
        let mag = LOG_LUT[((byte >> 3) & 0x07) as usize] as i32;
        let mul = LOG_LUT[(byte & 0x07) as usize] as i32;
        lut[i as usize] = pol * mag * mul;
        i += 1;
    }
    lut
};

/// Quantize a u8 value to the nearest 3-bit log-scale code.
#[inline]
const fn quantize(v: u8) -> u8 {
    if v == 0 {
        return 0;
    }
    // Thresholds are midpoints between adjacent LUT values
    if v <= 2 {
        return 1;
    }
    if v <= 10 {
        return 2;
    }
    if v <= 24 {
        return 3;
    }
    if v <= 48 {
        return 4;
    }
    if v <= 96 {
        return 5;
    }
    if v <= 191 {
        return 6;
    }
    7
}

/// Compact 1-byte signal: `[pol:2|mag:3|mul:3]`
///
/// Storage and wire format for ternary signals. Reduces memory from 3 bytes to 1 byte
/// per signal — a 67% reduction across the connectome.
///
/// ```text
/// Bit:  7  6  5  4  3  2  1  0
///       [pol ] [magnitude] [multiplier]
///        2 bit   3 bit       3 bit
/// ```
///
/// - Polarity: `00` = zero, `01` = +1, `10` = -1
/// - Magnitude/Multiplier: 3-bit codes into log-scale LUT `[0, 1, 4, 16, 32, 64, 128, 255]`
///
/// Unpacking to `(i8, u8, u8)` happens only at the soma for integration.
/// Shift-weighted integration keeps accumulators at i16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(transparent)]
pub struct PackedSignal(pub u8);

impl PackedSignal {
    /// Zero / silent packed signal.
    pub const ZERO: Self = Self(0);

    /// Maximum excitatory: pol=+1, mag=255, mul=255
    pub const MAX_POSITIVE: Self = Self(0b01_111_111);

    /// Maximum inhibitory: pol=-1, mag=255, mul=255
    pub const MAX_NEGATIVE: Self = Self(0b10_111_111);

    /// Pack from raw polarity, magnitude, and multiplier.
    ///
    /// Magnitude and multiplier are quantized to the nearest 3-bit log-scale code.
    #[inline]
    pub const fn pack(polarity: i8, magnitude: u8, multiplier: u8) -> Self {
        let p: u8 = if polarity > 0 {
            1
        } else if polarity < 0 {
            2
        } else {
            0
        };
        Self((p << 6) | (quantize(magnitude) << 3) | quantize(multiplier))
    }

    /// Pack from a [`Polarity`] enum.
    #[inline]
    pub const fn pack_typed(polarity: Polarity, magnitude: u8, multiplier: u8) -> Self {
        Self::pack(polarity.as_i8(), magnitude, multiplier)
    }

    /// Get the raw byte value.
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// Construct directly from a raw byte (no validation).
    #[inline]
    pub const fn from_raw(byte: u8) -> Self {
        Self(byte)
    }

    // -- Unpack components ---------------------------------------------------

    /// Extract polarity as i8.
    #[inline]
    pub const fn polarity(self) -> i8 {
        match self.0 >> 6 {
            1 => 1,
            2 => -1,
            _ => 0,
        }
    }

    /// Extract polarity as [`Polarity`] enum.
    #[inline]
    pub const fn get_polarity(self) -> Polarity {
        match self.0 >> 6 {
            1 => Polarity::Positive,
            2 => Polarity::Negative,
            _ => Polarity::Zero,
        }
    }

    /// Extract the 3-bit magnitude code (0–7).
    #[inline]
    pub const fn mag_code(self) -> u8 {
        (self.0 >> 3) & 0x07
    }

    /// Extract the 3-bit multiplier code (0–7).
    #[inline]
    pub const fn mul_code(self) -> u8 {
        self.0 & 0x07
    }

    /// Decode magnitude via LUT (0–255).
    #[inline]
    pub const fn magnitude(self) -> u8 {
        LOG_LUT[self.mag_code() as usize]
    }

    /// Decode multiplier via LUT (0–255).
    #[inline]
    pub const fn multiplier(self) -> u8 {
        LOG_LUT[self.mul_code() as usize]
    }

    // -- Current computation -------------------------------------------------

    /// Compute the true scalar current via precomputed LUT.
    ///
    /// Returns `polarity × magnitude × multiplier` as i32.
    /// Use this for operations that need the exact product (similarity, TERNARY_MATMUL,
    /// metabolic cost) — not for membrane integration (use [`shift_value`] instead).
    #[inline]
    pub const fn current(self) -> i32 {
        CURRENT_LUT[self.0 as usize]
    }

    /// Compute the shift-weighted integration value for membrane accumulation.
    ///
    /// Returns a signed i16 value: `polarity × (magnitude << shift)` where shift
    /// is determined by the multiplier code. Maximum: ±2040. Safe for i16 accumulators.
    #[inline]
    pub const fn shift_value(self) -> i16 {
        let pol = self.polarity();
        if pol == 0 {
            return 0;
        }

        let mag = self.magnitude() as i16;
        let shift = SHIFT_TABLE[self.mul_code() as usize];

        let adjusted = if shift > 0 {
            mag << (shift as u16)
        } else if shift < 0 {
            mag >> ((-shift) as u16)
        } else {
            mag
        };

        if pol > 0 {
            adjusted
        } else {
            -adjusted
        }
    }

    // -- Queries -------------------------------------------------------------

    /// Is this signal active (non-zero polarity and non-zero magnitude)?
    #[inline]
    pub const fn is_active(self) -> bool {
        let pol = self.0 >> 6;
        let mag_code = (self.0 >> 3) & 0x07;
        (pol == 1 || pol == 2) && mag_code > 0
    }

    /// Is this a positive / excitatory signal?
    #[inline]
    pub const fn is_positive(self) -> bool {
        (self.0 >> 6) == 1 && ((self.0 >> 3) & 0x07) > 0
    }

    /// Is this a negative / inhibitory signal?
    #[inline]
    pub const fn is_negative(self) -> bool {
        (self.0 >> 6) == 2 && ((self.0 >> 3) & 0x07) > 0
    }

    // -- Conversion to/from Signal -------------------------------------------

    /// Convert to the unpacked 3-byte [`Signal`] representation.
    ///
    /// Magnitude and multiplier are decoded through the log-scale LUT.
    #[inline]
    pub const fn to_signal(self) -> Signal {
        Signal {
            polarity: self.polarity(),
            magnitude: self.magnitude(),
            multiplier: self.multiplier(),
        }
    }

    /// Pack a [`Signal`] into compact 1-byte format.
    ///
    /// Lossy: magnitude and multiplier are quantized to 3-bit log-scale codes.
    #[inline]
    pub const fn from_signal(signal: &Signal) -> Self {
        Self::pack(signal.polarity, signal.magnitude, signal.multiplier)
    }
}

impl From<Signal> for PackedSignal {
    #[inline]
    fn from(s: Signal) -> Self {
        Self::from_signal(&s)
    }
}

impl From<PackedSignal> for Signal {
    #[inline]
    fn from(p: PackedSignal) -> Self {
        p.to_signal()
    }
}

impl core::fmt::Display for PackedSignal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Display uses the true current (LUT), same semantics as Signal
        let cur = self.current();
        if cur > 0 {
            write!(f, "+{}", cur)
        } else if cur < 0 {
            // current() already negative, display without extra minus
            write!(f, "{}", cur)
        } else {
            write!(f, "0")
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl core::fmt::Display for Signal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let eff = self.effective_magnitude();
        match self.polarity {
            p if p > 0 => write!(f, "+{}", eff),
            p if p < 0 => write!(f, "-{}", eff),
            _ => write!(f, "0"),
        }
    }
}

impl core::fmt::Display for Polarity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Polarity::Negative => write!(f, "-"),
            Polarity::Zero => write!(f, "0"),
            Polarity::Positive => write!(f, "+"),
        }
    }
}

// ---------------------------------------------------------------------------
// no_std f32 clamp helper (core doesn't have f32::clamp in all editions)
// ---------------------------------------------------------------------------

#[inline]
fn clamp_f32(val: f32, min: f32, max: f32) -> f32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn test_signal_size() {
        assert_eq!(core::mem::size_of::<Signal>(), 3);
    }

    #[test]
    fn test_signal_creation() {
        let zero = Signal::zero();
        assert!(!zero.is_active());

        let pos = Signal::positive(200);
        assert!(pos.is_positive());
        assert!(pos.is_active());
        assert_eq!(pos.multiplier, 1);
        assert_eq!(pos.effective_magnitude(), 200);

        let neg = Signal::negative(128);
        assert!(neg.is_negative());
        assert_eq!(neg.multiplier, 1);
    }

    #[test]
    fn test_amplified_creation() {
        let burst = Signal::positive_amplified(200, 100);
        assert!(burst.is_positive());
        assert_eq!(burst.magnitude, 200);
        assert_eq!(burst.multiplier, 100);
        assert_eq!(burst.effective_magnitude(), 20_000);

        let max = Signal::positive_amplified(255, 255);
        assert_eq!(max.effective_magnitude(), 65_025);

        let silenced = Signal::positive_amplified(200, 0);
        assert_eq!(silenced.effective_magnitude(), 0);
    }

    #[test]
    fn test_current() {
        let pos = Signal::positive_amplified(100, 50);
        assert_eq!(pos.current(), 5000);

        let neg = Signal::negative_amplified(100, 50);
        assert_eq!(neg.current(), -5000);

        let zero = Signal::zero();
        assert_eq!(zero.current(), 0);
    }

    #[test]
    fn test_multiplier_ops() {
        let sig = Signal::positive(150);
        assert_eq!(sig.multiplier, 1);

        let amped = sig.with_multiplier(200);
        assert_eq!(amped.magnitude, 150);
        assert_eq!(amped.multiplier, 200);
        assert_eq!(amped.effective_magnitude(), 30_000);

        let more = sig.amplify(50);
        assert_eq!(more.multiplier, 51); // 1 + 50

        let less = amped.attenuate(50);
        assert_eq!(less.multiplier, 150); // 200 - 50

        // Saturation
        let maxed = sig.amplify(255);
        assert_eq!(maxed.multiplier, 255); // saturates at 255

        let floored = sig.attenuate(255);
        assert_eq!(floored.multiplier, 0); // saturates at 0
    }

    #[test]
    fn test_from_signed() {
        let from_pos = Signal::from_signed(0.75);
        assert_eq!(from_pos.polarity, 1);
        assert_eq!(from_pos.multiplier, 1);

        let from_neg = Signal::from_signed(-0.5);
        assert_eq!(from_neg.polarity, -1);

        let from_zero = Signal::from_signed(0.005);
        assert_eq!(from_zero.polarity, 0);
    }

    #[test]
    fn test_decay() {
        let mut signal = Signal::positive(200);
        signal.decay(0.5);
        assert_eq!(signal.magnitude, 100);
        for _ in 0..10 {
            signal.decay(0.5);
        }
        assert!(!signal.is_active());
        assert_eq!(signal.polarity, 0);
    }

    #[test]
    fn test_decay_preserves_multiplier() {
        let mut signal = Signal::positive_amplified(200, 100);
        signal.decay(0.5);
        assert_eq!(signal.magnitude, 100);
        assert_eq!(signal.multiplier, 100); // multiplier unchanged
    }

    #[test]
    fn test_add() {
        let a = Signal::positive(100);
        let b = Signal::positive(100);
        let sum = a.add(&b);
        assert!(sum.is_positive());

        let c = Signal::negative(100);
        let cancel = a.add(&c);
        assert!(!cancel.is_active() || cancel.magnitude < 10);
    }

    #[test]
    fn test_polarity_enum() {
        assert_eq!(Polarity::Negative.as_i8(), -1);
        assert_eq!(Polarity::Zero.as_i8(), 0);
        assert_eq!(Polarity::Positive.as_i8(), 1);
        assert_eq!(Polarity::from_i8(2), None);
    }

    #[test]
    fn test_step_toward() {
        let signal = Signal::positive(100);
        let target = Signal::positive(200);
        let result = signal.step_toward(&target);
        assert_eq!(result.magnitude, 101);
    }

    #[test]
    fn test_step_toward_preserves_multiplier() {
        let signal = Signal::positive(100).with_multiplier(50);
        let target = Signal::positive(200);
        let result = signal.step_toward(&target);
        assert_eq!(result.magnitude, 101);
        assert_eq!(result.multiplier, 50); // preserved from source
    }

    #[test]
    fn test_new_checked() {
        assert!(Signal::new_checked(1, 100).is_some());
        assert!(Signal::new_checked(2, 100).is_none());
    }

    #[test]
    fn test_display() {
        extern crate alloc;
        use alloc::format;
        // Display shows effective magnitude (magnitude × multiplier)
        assert_eq!(format!("{}", Signal::positive(42)), "+42"); // 42 × 1
        assert_eq!(format!("{}", Signal::negative(10)), "-10"); // 10 × 1
        assert_eq!(format!("{}", Signal::zero()), "0");
        assert_eq!(format!("{}", Signal::positive_amplified(10, 100)), "+1000");
        // 10 × 100
    }

    #[test]
    fn test_constants() {
        assert_eq!(Signal::ZERO.polarity, 0);
        assert_eq!(Signal::ZERO.magnitude, 0);
        assert_eq!(Signal::ZERO.multiplier, 0);

        assert_eq!(Signal::MAX_POSITIVE.polarity, 1);
        assert_eq!(Signal::MAX_POSITIVE.magnitude, 255);
        assert_eq!(Signal::MAX_POSITIVE.multiplier, 255);
        assert_eq!(Signal::MAX_POSITIVE.effective_magnitude(), 65_025);

        assert_eq!(Signal::MAX_NEGATIVE.polarity, -1);
        assert_eq!(Signal::MAX_NEGATIVE.magnitude, 255);
        assert_eq!(Signal::MAX_NEGATIVE.multiplier, 255);
    }

    // -- PackedSignal tests --------------------------------------------------

    #[test]
    fn test_packed_signal_size() {
        assert_eq!(core::mem::size_of::<PackedSignal>(), 1);
    }

    #[test]
    fn test_packed_zero() {
        let z = PackedSignal::ZERO;
        assert_eq!(z.0, 0);
        assert_eq!(z.polarity(), 0);
        assert_eq!(z.magnitude(), 0);
        assert_eq!(z.multiplier(), 0);
        assert_eq!(z.current(), 0);
        assert_eq!(z.shift_value(), 0);
        assert!(!z.is_active());
    }

    #[test]
    fn test_packed_polarity_encoding() {
        let pos = PackedSignal::pack(1, 100, 50);
        assert_eq!(pos.polarity(), 1);
        assert!(pos.is_positive());
        assert!(!pos.is_negative());

        let neg = PackedSignal::pack(-1, 100, 50);
        assert_eq!(neg.polarity(), -1);
        assert!(!neg.is_positive());
        assert!(neg.is_negative());

        let zero = PackedSignal::pack(0, 100, 50);
        assert_eq!(zero.polarity(), 0);
        assert!(!zero.is_active());
    }

    #[test]
    fn test_packed_magnitude_lut() {
        // Verify each code maps correctly
        for code in 0..8u8 {
            let packed = PackedSignal::from_raw(0b01_000_000 | (code << 3));
            assert_eq!(packed.mag_code(), code);
            assert_eq!(packed.magnitude(), LOG_LUT[code as usize]);
        }
    }

    #[test]
    fn test_packed_multiplier_lut() {
        for code in 0..8u8 {
            let packed = PackedSignal::from_raw(0b01_000_000 | code);
            assert_eq!(packed.mul_code(), code);
            assert_eq!(packed.multiplier(), LOG_LUT[code as usize]);
        }
    }

    #[test]
    fn test_packed_current_lut() {
        // Verify CURRENT_LUT matches manual computation for every byte value
        for byte in 0..=255u8 {
            let p = PackedSignal::from_raw(byte);
            let manual = p.polarity() as i32 * p.magnitude() as i32 * p.multiplier() as i32;
            assert_eq!(
                p.current(),
                manual,
                "CURRENT_LUT mismatch at byte 0x{:02X}",
                byte
            );
        }
    }

    #[test]
    fn test_packed_max_values() {
        let max_pos = PackedSignal::MAX_POSITIVE;
        assert_eq!(max_pos.polarity(), 1);
        assert_eq!(max_pos.magnitude(), 255);
        assert_eq!(max_pos.multiplier(), 255);
        assert_eq!(max_pos.current(), 65_025);

        let max_neg = PackedSignal::MAX_NEGATIVE;
        assert_eq!(max_neg.polarity(), -1);
        assert_eq!(max_neg.current(), -65_025);
    }

    #[test]
    fn test_packed_shift_value_range() {
        // Maximum shift_value should fit i16 (max ±2040)
        let max = PackedSignal::MAX_POSITIVE;
        let sv = max.shift_value();
        assert!(sv <= 2040, "shift_value {} exceeds 2040", sv);
        assert!(sv > 0);

        let min = PackedSignal::MAX_NEGATIVE;
        let sv_neg = min.shift_value();
        assert!(sv_neg >= -2040, "shift_value {} below -2040", sv_neg);
        assert!(sv_neg < 0);

        // Zero polarity always yields 0
        let zero = PackedSignal::pack(0, 255, 255);
        assert_eq!(zero.shift_value(), 0);
    }

    #[test]
    fn test_packed_shift_attenuate_amplify() {
        // Low multiplier should attenuate
        let quiet = PackedSignal::pack(1, 128, 1); // mul code 1 → >>2
        let sv = quiet.shift_value();
        let raw_mag = quiet.magnitude() as i16;
        assert!(
            sv < raw_mag,
            "low mul should attenuate: shift={} vs raw={}",
            sv,
            raw_mag
        );

        // High multiplier should amplify
        let burst = PackedSignal::pack(1, 128, 255); // mul code 7 → <<3
        let sv_burst = burst.shift_value();
        assert!(
            sv_burst > raw_mag,
            "high mul should amplify: shift={} vs raw={}",
            sv_burst,
            raw_mag
        );
    }

    #[test]
    fn test_packed_quantization_roundtrip() {
        // Quantization is lossy but consistent
        let cases: &[(i8, u8, u8)] = &[
            (1, 0, 0),
            (1, 1, 1),
            (-1, 50, 30),
            (1, 200, 100),
            (1, 255, 255),
            (0, 128, 64),
        ];
        for &(pol, mag, mul) in cases {
            let packed = PackedSignal::pack(pol, mag, mul);
            // Polarity is always exact
            assert_eq!(
                packed.polarity(),
                pol,
                "polarity mismatch for ({}, {}, {})",
                pol,
                mag,
                mul
            );
            // Magnitude and multiplier are quantized — just check they're in LUT
            assert!(LOG_LUT.contains(&packed.magnitude()));
            assert!(LOG_LUT.contains(&packed.multiplier()));
        }
    }

    #[test]
    fn test_packed_signal_conversion() {
        let signal = Signal::positive_amplified(200, 100);
        let packed = PackedSignal::from(signal);
        let back = Signal::from(packed);

        // Polarity survives exactly
        assert_eq!(back.polarity, signal.polarity);
        // Magnitude and multiplier are quantized (lossy)
        assert!(LOG_LUT.contains(&back.magnitude));
        assert!(LOG_LUT.contains(&back.multiplier));
    }

    #[test]
    fn test_packed_from_signal_method() {
        let s = Signal::negative(50);
        let p = PackedSignal::from_signal(&s);
        assert_eq!(p.polarity(), -1);
        assert!(p.is_negative());

        let roundtrip = p.to_signal();
        assert_eq!(roundtrip.polarity, -1);
    }

    #[test]
    fn test_packed_typed_polarity() {
        let p = PackedSignal::pack_typed(Polarity::Positive, 100, 64);
        assert_eq!(p.get_polarity(), Polarity::Positive);

        let n = PackedSignal::pack_typed(Polarity::Negative, 100, 64);
        assert_eq!(n.get_polarity(), Polarity::Negative);

        let z = PackedSignal::pack_typed(Polarity::Zero, 100, 64);
        assert_eq!(z.get_polarity(), Polarity::Zero);
    }

    #[test]
    fn test_packed_display() {
        extern crate alloc;
        use alloc::format;
        assert_eq!(format!("{}", PackedSignal::ZERO), "0");
        assert_eq!(format!("{}", PackedSignal::MAX_POSITIVE), "+65025");
        assert_eq!(format!("{}", PackedSignal::MAX_NEGATIVE), "-65025");
    }
}
