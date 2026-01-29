//! # ternary-signal
//!
//! The fundamental unit of ternary neural communication: `Signal = polarity × magnitude`
//!
//! Compact 2-byte representation combining:
//! - **Polarity**: excitatory (+1), inhibitory (-1), or silent (0)
//! - **Magnitude**: intensity 0–255 (maps to 0.0–1.0)
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

/// Signal: polarity + magnitude (`s = p × m`)
///
/// The fundamental unit of neural communication.
/// Compact 2-byte `#[repr(C)]` representation:
/// - `polarity`: -1 (inhibited), 0 (neutral), +1 (excited)
/// - `magnitude`: 0–255 (maps to 0.0–1.0 intensity)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C)]
pub struct Signal {
    /// Polarity: -1 (inhibited), 0 (neutral), +1 (excited)
    pub polarity: i8,
    /// Magnitude: 0–255 (intensity, maps to 0.0–1.0)
    pub magnitude: u8,
}

impl Signal {
    // -- Constants -----------------------------------------------------------

    /// Zero signal (no activity)
    pub const ZERO: Self = Self { polarity: 0, magnitude: 0 };
    /// Maximum positive signal
    pub const MAX_POSITIVE: Self = Self { polarity: 1, magnitude: 255 };
    /// Maximum negative signal
    pub const MAX_NEGATIVE: Self = Self { polarity: -1, magnitude: 255 };

    // -- Constructors --------------------------------------------------------

    /// Create a zero / neutral signal
    #[inline]
    pub const fn zero() -> Self { Self::ZERO }

    /// Create a positive (excitatory) signal
    #[inline]
    pub const fn positive(magnitude: u8) -> Self {
        Self { polarity: 1, magnitude }
    }

    /// Create a negative (inhibitory) signal
    #[inline]
    pub const fn negative(magnitude: u8) -> Self {
        Self { polarity: -1, magnitude }
    }

    /// Create from [`Polarity`] enum and magnitude (type-safe)
    #[inline]
    pub const fn with_polarity(polarity: Polarity, magnitude: u8) -> Self {
        Self { polarity: polarity.as_i8(), magnitude }
    }

    /// Create from raw `i8` polarity and magnitude (**unchecked**)
    #[inline]
    pub const fn new(polarity: i8, magnitude: u8) -> Self {
        Self { polarity, magnitude }
    }

    /// Create from raw `i8` polarity with validation
    #[inline]
    pub const fn new_checked(polarity: i8, magnitude: u8) -> Option<Self> {
        match Polarity::from_i8(polarity) {
            Some(_) => Some(Self { polarity, magnitude }),
            None => None,
        }
    }

    /// Create from separate float components
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
        Self { polarity, magnitude }
    }

    /// Create from a single signed float (-1.0 to 1.0)
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
        Self { polarity, magnitude }
    }

    /// Create from a signed `i32`
    #[inline]
    pub fn from_signed_i32(value: i32) -> Self {
        if value == 0 {
            Self::ZERO
        } else if value > 0 {
            Self { polarity: 1, magnitude: (if value > 255 { 255 } else { value }) as u8 }
        } else {
            let abs = if value == i32::MIN { 255 } else { -value };
            Self { polarity: -1, magnitude: (if abs > 255 { 255 } else { abs }) as u8 }
        }
    }

    /// Create from a signed `i16`
    #[inline]
    pub fn from_i16(value: i16) -> Self {
        if value == 0 {
            Self::ZERO
        } else if value > 0 {
            Self { polarity: 1, magnitude: (if value > 255 { 255 } else { value }) as u8 }
        } else {
            let abs = -(value as i32);
            Self { polarity: -1, magnitude: (if abs > 255 { 255 } else { abs }) as u8 }
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
        if magnitude == 0 { Self::ZERO } else { Self::positive(magnitude) }
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

    /// Get as signed `i32` (`polarity × magnitude`)
    #[inline]
    pub fn as_signed_i32(&self) -> i32 {
        self.polarity as i32 * self.magnitude as i32
    }

    /// Get magnitude as float (0.0–1.0)
    #[inline]
    pub fn magnitude_f32(&self) -> f32 {
        self.magnitude as f32 / 255.0
    }

    /// Get as signed float (-1.0 to 1.0)
    #[inline]
    pub fn as_signed_f32(&self) -> f32 {
        self.polarity as f32 * self.magnitude_f32()
    }

    /// Convert to u8 bipolar (128 = baseline)
    #[inline]
    pub fn to_u8_bipolar(&self) -> u8 {
        let signed = self.polarity as i16 * self.magnitude as i16;
        let result = signed + 128;
        (if result < 0 { 0 } else if result > 255 { 255 } else { result }) as u8
    }

    /// Convert to spike rate (Hz)
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

    // -- Arithmetic ----------------------------------------------------------

    /// Add two signals (same polarity adds, opposite cancels)
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let combined = self.as_signed_f32() + other.as_signed_f32();
        Self::from_signed(combined)
    }

    /// Scale magnitude by a factor
    #[inline]
    pub fn scale(&self, factor: f32) -> Self {
        let new_mag = clamp_f32(self.magnitude as f32 * factor, 0.0, 255.0) as u8;
        Self {
            polarity: if new_mag > 0 { self.polarity } else { 0 },
            magnitude: new_mag,
        }
    }

    /// Apply decay to magnitude (in-place, for temporal fields)
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

    /// Step toward target by one unit
    #[inline]
    pub fn step_toward(&self, target: &Self) -> Self {
        self.step_toward_by(target, 1)
    }

    /// Step toward target by specified delta
    #[inline]
    pub fn step_toward_by(&self, target: &Self, delta: u8) -> Self {
        let current = self.polarity as i16 * self.magnitude as i16;
        let target_val = target.polarity as i16 * target.magnitude as i16;
        let diff = target_val - current;

        if diff.unsigned_abs() <= delta as u16 {
            *target
        } else if diff > 0 {
            Self::from_signed_i32((current + delta as i16) as i32)
        } else {
            Self::from_signed_i32((current - delta as i16) as i32)
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
            if raw < 1 { 1 } else { raw }
        };
        self.step_toward_by(target, delta as u8)
    }

    /// Check if this signal has reached the target (within tolerance)
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
// Display
// ---------------------------------------------------------------------------

impl core::fmt::Display for Signal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.polarity {
            p if p > 0 => write!(f, "+{}", self.magnitude),
            p if p < 0 => write!(f, "-{}", self.magnitude),
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
    if val < min { min } else if val > max { max } else { val }
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
        assert_eq!(core::mem::size_of::<Signal>(), 2);
    }

    #[test]
    fn test_signal_creation() {
        let zero = Signal::zero();
        assert!(!zero.is_active());

        let pos = Signal::positive(200);
        assert!(pos.is_positive());
        assert!(pos.is_active());

        let neg = Signal::negative(128);
        assert!(neg.is_negative());
    }

    #[test]
    fn test_from_signed() {
        let from_pos = Signal::from_signed(0.75);
        assert_eq!(from_pos.polarity, 1);

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
    fn test_add() {
        let a = Signal::positive(100);
        let b = Signal::positive(100);
        let sum = a.add(&b);
        assert!(sum.is_positive());
        assert!(sum.magnitude > 150);

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
    fn test_new_checked() {
        assert!(Signal::new_checked(1, 100).is_some());
        assert!(Signal::new_checked(2, 100).is_none());
    }

    #[test]
    fn test_display() {
        extern crate alloc;
        use alloc::format;
        assert_eq!(format!("{}", Signal::positive(42)), "+42");
        assert_eq!(format!("{}", Signal::negative(10)), "-10");
        assert_eq!(format!("{}", Signal::zero()), "0");
    }
}
