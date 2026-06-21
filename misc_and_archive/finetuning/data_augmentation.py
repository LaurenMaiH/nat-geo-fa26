import cv2
import numpy as np
from pathlib import Path
import random


def augment_images(
    input_dir: str | Path,
    output_dir: str | Path,
    augmentations_per_image: int = 5,
    seed: int = 42,
):
    """
    Generate augmented copies of every image in input_dir and save to output_dir.

    Augmentations applied (randomly sampled each time):
        - Horizontal / vertical flip
        - Rotation (-30 to +30 degrees)
        - Brightness & contrast jitter
        - Gaussian blur
        - Gaussian noise
        - Zoom / crop
        - Horizontal shear
    """
    random.seed(seed)
    np.random.seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in extensions]

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Augmenting {len(images)} image(s), {augmentations_per_image} copies each "
          f"→ {len(images) * augmentations_per_image} new images")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read {img_path.name}, skipping")
            continue

        h, w = img.shape[:2]

        for i in range(augmentations_per_image):
            aug = img.copy()

            # --- Flip ---
            flip = random.choice([-1, 0, 1, None])
            if flip is not None:
                aug = cv2.flip(aug, flip)

            # --- Rotation ---
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # --- Brightness / contrast ---
            alpha = random.uniform(0.6, 1.4)   # contrast
            beta  = random.randint(-40, 40)     # brightness
            aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

            # --- Gaussian blur ---
            if random.random() < 0.5:
                ksize = random.choice([3, 5, 7])
                aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

            # --- Gaussian noise ---
            if random.random() < 0.5:
                noise = np.random.normal(0, random.uniform(5, 25), aug.shape).astype(np.int16)
                aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # --- Zoom / crop ---
            if random.random() < 0.5:
                scale = random.uniform(0.75, 0.95)
                cx, cy = w // 2, h // 2
                crop_w, crop_h = int(w * scale), int(h * scale)
                x1 = max(cx - crop_w // 2, 0)
                y1 = max(cy - crop_h // 2, 0)
                x2 = min(x1 + crop_w, w)
                y2 = min(y1 + crop_h, h)
                aug = cv2.resize(aug[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)

            out_name = f"{img_path.stem}_aug{i:03d}{img_path.suffix}"
            cv2.imwrite(str(output_dir / out_name), aug)

    print(f"Done. Saved to {output_dir}")


# Example usage
CROPS_DIR = Path(__file__).parent / "crops"
CROPS_AUGMENTED_DIR = Path(__file__).parent / "crops_augmented"

for species_dir in sorted(CROPS_DIR.iterdir()):
    if species_dir.is_dir():
        augment_images(
            input_dir=species_dir,
            output_dir=CROPS_AUGMENTED_DIR / species_dir.name,
            augmentations_per_image=10,
        )