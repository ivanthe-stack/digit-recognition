import os
import struct
import gzip
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.ndimage import map_coordinates

# --- CONFIG ---
augmentations_per_image = 10
max_shift = 5
batch_size = 2000  # adjust for available RAM
# ---------------

# --- FORCE CPU ---
jax.config.update("jax_platform_name", "cpu")

# --- IDX UTILITIES ---
def _read_idx_images(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        data = f.read()
    magic, num, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"Bad magic number for images: {magic}")
    arr = jnp.frombuffer(data, dtype=jnp.uint8, offset=16)
    return arr.reshape((num, rows * cols))

def _read_idx_labels(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        data = f.read()
    magic, num = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"Bad magic number for labels: {magic}")
    return jnp.frombuffer(data, dtype=jnp.uint8, offset=8)

def write_idx_images(path, arr):
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, arr.shape[0], 28, 28))
        f.write(memoryview(arr.astype(jnp.uint8)))

def write_idx_labels(path, arr):
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, arr.shape[0]))
        f.write(memoryview(arr.astype(jnp.uint8)))

# --- AUGMENTATION (JAX + JIT + VMAP) ---
@jax.jit
def augment_image(image: jnp.ndarray, shift_y: float, shift_x: float) -> jnp.ndarray:
    """Shifts one image by (shift_y, shift_x) using bilinear interpolation."""
    image = image.reshape((28, 28))
    y, x = jnp.meshgrid(jnp.arange(28), jnp.arange(28), indexing="ij")
    y_shifted = jnp.clip(y - shift_y, 0, 27)
    x_shifted = jnp.clip(x - shift_x, 0, 27)
    return map_coordinates(image, [y_shifted, x_shifted], order=1, mode="constant", cval=0.0).reshape(-1)

@jax.jit
def batch_augment(images, shift_y, shift_x):
    return jax.vmap(augment_image)(images, shift_y, shift_x)

def generate_shifts(key, n, max_shift):
    """Generate random shifts for n images."""
    key_y, key_x = random.split(key)
    shift_y = random.uniform(key_y, (n,), minval=-max_shift, maxval=max_shift)
    shift_x = random.uniform(key_x, (n,), minval=-max_shift, maxval=max_shift)
    return shift_y, shift_x

# --- MAIN ---
def prepare_augmented_data():
    IMG_PATH = "data/train-images-idx3-ubyte"
    LBL_PATH = "data/train-labels-idx1-ubyte"

    img_path_to_read = IMG_PATH + ".gz" if os.path.exists(IMG_PATH + ".gz") else IMG_PATH
    lbl_path_to_read = LBL_PATH + ".gz" if os.path.exists(LBL_PATH + ".gz") else LBL_PATH

    if not os.path.exists(img_path_to_read):
        print("Error: Missing MNIST image file.")
        return

    with open(img_path_to_read, "rb") as f:
        header = gzip.open(f).read(16) if img_path_to_read.endswith(".gz") else f.read(16)
    _, num_images, _, _ = struct.unpack(">IIII", header)
    if num_images > 60000:
        print(f"Already augmented ({num_images} images). Skipping.")
        return

    print("Reading MNIST data...")
    orig_images = _read_idx_images(img_path_to_read)
    orig_labels = _read_idx_labels(lbl_path_to_read)
    print(f"Found {len(orig_images)} original images.")

    images_f32 = orig_images.astype(jnp.float32) / 255.0
    total_aug = len(images_f32) * augmentations_per_image
    print(f"⚙️ Generating {total_aug} augmented images...")

    key = random.PRNGKey(0)
    augmented_chunks = []
    augmented_labels = []

    for i in range(0, len(images_f32), batch_size):
        batch = images_f32[i : i + batch_size]
        labels = orig_labels[i : i + batch_size]

        for _ in range(augmentations_per_image):
            key, subkey = random.split(key)
            shift_y, shift_x = generate_shifts(subkey, len(batch), max_shift)
            aug_batch = batch_augment(batch, shift_y, shift_x)
            augmented_chunks.append(aug_batch)
            augmented_labels.append(labels)

        print(f"Processed {i + len(batch):>6}/{len(images_f32)} images...")

    augmented_images = jnp.concatenate(augmented_chunks)
    augmented_labels = jnp.concatenate(augmented_labels)

    # Combine: originals first, augmented last
    combined_images = jnp.concatenate((orig_images, (augmented_images * 255).astype(jnp.uint8)))
    combined_labels = jnp.concatenate((orig_labels, augmented_labels))

    print(f"Writing {len(combined_images)} total images to original MNIST files...")
    write_idx_images(IMG_PATH, combined_images)
    write_idx_labels(LBL_PATH, combined_labels)
    print("Done. Originals first, then augmented. Files overwritten.")

if __name__ == "__main__":
    prepare_augmented_data()
