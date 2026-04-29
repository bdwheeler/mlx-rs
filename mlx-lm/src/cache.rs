use mlx_rs::{error::Exception, ops::concatenate_axis, transforms::eval, Array};

// TODO: somehow move quantized methods to a separate trait?
pub trait KeyValueCache {
    fn is_quantized(&self) -> bool {
        false
    }

    /// Returns the group size used for quantization. `None` if not quantized.
    fn group_size(&self) -> Option<i32> {
        None
    }

    /// Returns the number of bits used for quantization. `None` if not quantized.
    fn bits(&self) -> Option<i32> {
        None
    }

    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array)
        -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn is_quantized(&self) -> bool {
        T::is_quantized(self)
    }

    fn group_size(&self) -> Option<i32> {
        T::group_size(self)
    }

    fn bits(&self) -> Option<i32> {
        T::bits(self)
    }

    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

/// KV cache backed by concatenation: each call to `update_and_fetch` builds a
/// fresh contiguous K/V tensor by concatenating the new K/V with the existing.
/// Returns the concatenated tensor directly (which is contiguous because
/// concatenate produces a fresh array).
#[derive(Debug, Clone, Default)]
pub struct ConcatKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl ConcatKeyValueCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Trim the cache to retain only the first `new_offset` positions along the
    /// sequence axis. Used after tree-shaped speculative verification to discard
    /// KV state for rejected draft branches, leaving only the accepted prefix.
    ///
    /// No-op if `new_offset >= self.offset`. Setting `new_offset <= 0` clears the
    /// cache entirely.
    pub fn trim_to(&mut self, new_offset: i32) {
        use mlx_rs::ops::indexing::IndexOp;
        if new_offset >= self.offset {
            return;
        }
        if new_offset <= 0 {
            self.keys = None;
            self.values = None;
            self.offset = 0;
            return;
        }
        // Slice the seq_len axis (-2). qwen3/llama store K/V as 4D
        // [B, n_kv_heads, seq_len, head_dim] after the [0,2,1,3] transpose;
        // 3D [n_kv_heads, seq_len, head_dim] is also possible for caches built
        // outside the model. Dispatch on rank — mlx-rs's index tuples are
        // length-fixed and a 3-axis tuple silently slices the wrong axis on
        // a 4D array.
        let trim = |a: &Array, new_offset: i32| -> Array {
            match a.shape().len() {
                3 => a.index((.., ..new_offset, ..)),
                4 => a.index((.., .., ..new_offset, ..)),
                r => panic!("ConcatKeyValueCache::trim_to: unsupported KV rank {r}"),
            }
        };
        if let Some(k) = self.keys.as_ref() {
            self.keys = Some(trim(k, new_offset));
        }
        if let Some(v) = self.values.as_ref() {
            self.values = Some(trim(v, new_offset));
        }
        self.offset = new_offset;
    }
}

impl KeyValueCache for ConcatKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match (self.keys.take(), self.values.take()) {
            (Some(k), Some(v)) => {
                self.keys = Some(concatenate_axis(&[k, keys], -2)?);
                self.values = Some(concatenate_axis(&[v, values], -2)?);
                eval(std::slice::from_ref(self.keys.as_ref().unwrap()))
                    .map_err(|e| Exception::custom(format!("eval KV keys: {e}")))?;
                eval(std::slice::from_ref(self.values.as_ref().unwrap()))
                    .map_err(|e| Exception::custom(format!("eval KV values: {e}")))?;
            }
            _ => {
                self.keys = Some(keys);
                self.values = Some(values);
            }
        }
        let shape = self.keys.as_ref().unwrap().shape();
        self.offset = shape[shape.len() - 2];
        Ok((
            self.keys.clone().unwrap(),
            self.values.clone().unwrap(),
        ))
    }
}

/// TODO: A generic KV Cache
pub struct DefaultKeyValueCache {}
