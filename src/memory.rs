use std::ops::{Deref, DerefMut};

use zeroize::Zeroize;

pub struct WipeOnDrop<T: Zeroize>(T);

impl<T: Zeroize> WipeOnDrop<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: Zeroize> Deref for WipeOnDrop<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Zeroize> DerefMut for WipeOnDrop<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Zeroize> Drop for WipeOnDrop<T> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
