#!/usr/bin/env python3
"""
cnn_numeros.py
Red neuronal convolucional (CNN) para clasificar los números del 1 al 75
construidos a partir de imágenes de dígitos 28x28.

Las imágenes de entrada tienen dos posibles tamaños:
  - 28 x 28 px  →  números 1-9  (1 dígito)
  - 64 x 28 px  →  números 10-75 (2 dígitos, gap=8px por defecto)

La red redimensiona todo a un tamaño fijo (64x28) en el preprocesamiento,
de modo que un único modelo maneja ambos casos.

Estructura esperada de la carpeta de datos:
    train/
        1/   1_000001.png  1_000002.png  ...
        2/   2_000001.png  ...
        ...
        75/  75_000001.png ...

Uso:
    # Entrenamiento
    python cnn_numeros.py train --datos ruta/a/train/ --epocas 30

    # Predicción sobre una imagen
    python cnn_numeros.py predict --imagen ruta/imagen.png --modelo modelo.pth

    # Evaluación sobre una carpeta
    python cnn_numeros.py eval --datos ruta/a/train/ --modelo modelo.pth

Dependencias:
    pip install torch torchvision pillow scikit-learn matplotlib
"""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ─────────────────────────────────────────
# Hiperparámetros por defecto
# ─────────────────────────────────────────
IMG_W, IMG_H = 64, 64          # tamaño al que se normaliza toda imagen
NUM_CLASES   = 75              # números 1-75
BATCH_SIZE   = 16
LR           = 1e-3
EPOCAS       = 30
PATIENCE     = 10              # épocas sin mejora antes de detener
SPLIT_VAL    = 0.2             # 20% validación
SEMILLA      = 42
MODELO_PATH  = "cnnModel.pth"  # define the name of the model


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class NumerosDataset(Dataset):
    """
    Carga imágenes desde la estructura:
        raiz/
            1/   1_000001.png  1_000002.png  ...
            2/   2_000001.png  ...
            ...
            75/  75_000001.png ...

    El nombre de la subcarpeta es el número (1-75) -> clase 0-indexada.
    """

    def __init__(self, carpeta: Path, transform=None):
        self.muestras = []
        self.transform = transform

        extensiones = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

        def sort_key(p):
            try:
                return int(p.name)
            except ValueError:
                return 999

        for subcarpeta in sorted(carpeta.iterdir(), key=sort_key):
            if not subcarpeta.is_dir():
                continue
            try:
                numero = int(subcarpeta.name)
            except ValueError:
                continue
            if not (1 <= numero <= 75):
                continue

            clase = numero - 1  # 0-indexado
            for img_ruta in sorted(subcarpeta.iterdir()):
                if img_ruta.suffix.lower() in extensiones:
                    self.muestras.append((img_ruta, clase))

        if not self.muestras:
            raise RuntimeError(
                f"No se encontraron imágenes en: {carpeta}\n"
                f"Estructura esperada: {carpeta}/1/, {carpeta}/2/, ... {carpeta}/75/"
            )

        clases_encontradas = len({c for _, c in self.muestras})
        print(f"  Clases encontradas : {clases_encontradas}/75")
        print(f"  Total imágenes     : {len(self.muestras)}")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        ruta, etiqueta = self.muestras[idx]
        img = Image.open(ruta).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, etiqueta


def get_transform(augment: bool = False):
    ops = [
        transforms.Resize((IMG_H, IMG_W)),
        transforms.Grayscale(num_output_channels=1),
    ]
    if augment:
        ops += [
            # Aumentación agresiva para dataset pequeño (~70 fotos/clase)
            # Rotación: los números en bolas pueden venir girados
            transforms.RandomAffine(
                degrees=15,
                translate=(0.10, 0.10),
                scale=(0.85, 1.15),
                shear=5,
                fill=0,           # fondo negro (bolas de bingo)
            ),
            # Perspectiva: simula ángulos de cámara distintos
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=0),
            # Brillo/contraste/saturación: variaciones de iluminación
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            # Blur leve: simula foco imperfecto de cámara
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            # Volteo horizontal: el 1, 8 y 0 son simétricos; ayuda en general
            transforms.RandomHorizontalFlip(p=0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # [-1, 1]
    ]
    return transforms.Compose(ops)


# ─────────────────────────────────────────
# Arquitectura CNN
# ─────────────────────────────────────────
class CNN_Numeros(nn.Module):
    """
    CNN para clasificar números 1-75 en imágenes 1×64×64 (bolas de bingo).

    Diseñada para imágenes cuadradas con fondo circular oscuro y número
    centrado, sin importar si tiene 1 o 2 dígitos.

    Entrada : 1 × 64 × 64
    Bloque 1: 32 filtros  → 32 × 32 × 32  (rasgos locales: bordes, curvas)
    Bloque 2: 64 filtros  → 64 × 16 × 16  (rasgos medios: partes de dígito)
    Bloque 3: 128 filtros →128 ×  8 ×  8  (rasgos globales: forma del número)
    Bloque 4: 256 filtros →256 ×  4 ×  4  (contexto completo)
    GAP     : 256                          (Global Average Pooling)
    FC      : 256 → 128 → 75 clases
    """

    def __init__(self, num_clases: int = NUM_CLASES):
        super().__init__()

        # ── Bloque 1: rasgos locales  (64×64 → 32×32) ────────────────
        self.bloque1 = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 32×32
            nn.Dropout2d(0.20),
        )

        # ── Bloque 2: rasgos medios   (32×32 → 16×16) ────────────────
        self.bloque2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16×16
            nn.Dropout2d(0.25),
        )

        # ── Bloque 3: rasgos globales (16×16 → 8×8) ──────────────────
        self.bloque3 = nn.Sequential(
            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 8×8
            nn.Dropout2d(0.25),
        )

        # ── Bloque 4: contexto completo (8×8 → 4×4) ──────────────────
        self.bloque4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 4×4
            nn.Dropout2d(0.25),
        )

        # ── Global Average Pooling: 256×4×4 → 256 ────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Cabeza clasificadora ──────────────────────────────────────
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_clases),
        )

    def forward(self, x):
        x = self.bloque1(x)
        x = self.bloque2(x)
        x = self.bloque3(x)
        x = self.bloque4(x)
        x = self.gap(x)
        return self.clasificador(x)


# ─────────────────────────────────────────
# Utilidades de entrenamiento
# ─────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def entrenar_epoca(modelo, loader, criterio, optimizador, device):
    modelo.train()
    total_loss, correctos, total = 0.0, 0, 0
    for imgs, etiquetas in loader:
        imgs, etiquetas = imgs.to(device), etiquetas.to(device)
        optimizador.zero_grad()
        salidas = modelo(imgs)
        loss = criterio(salidas, etiquetas)
        loss.backward()
        optimizador.step()
        total_loss += loss.item() * imgs.size(0)
        correctos  += (salidas.argmax(1) == etiquetas).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correctos / total


@torch.no_grad()
def evaluar(modelo, loader, criterio, device):
    modelo.eval()
    total_loss, correctos, total = 0.0, 0, 0
    for imgs, etiquetas in loader:
        imgs, etiquetas = imgs.to(device), etiquetas.to(device)
        salidas = modelo(imgs)
        loss = criterio(salidas, etiquetas)
        total_loss += loss.item() * imgs.size(0)
        correctos  += (salidas.argmax(1) == etiquetas).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correctos / total


# ─────────────────────────────────────────
# Comando: train
# ─────────────────────────────────────────
def cmd_train(args):
    torch.manual_seed(SEMILLA)
    random.seed(SEMILLA)
    device = get_device()
    print(f"Dispositivo : {device}")

    carpeta = Path(args.datos)
    dataset_full = NumerosDataset(carpeta, transform=get_transform(augment=True))
    print(f"Imágenes    : {len(dataset_full)}")

    n_val   = max(1, int(len(dataset_full) * SPLIT_VAL))
    n_train = len(dataset_full) - n_val
    train_set, val_set = random_split(
        dataset_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEMILLA)
    )
    # Validación sin aumentación
    val_set.dataset = NumerosDataset(carpeta, transform=get_transform(augment=False))

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, num_workers=0)

    modelo     = CNN_Numeros(num_clases=NUM_CLASES).to(device)
    criterio   = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizador = optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=args.epocas)

    mejor_val_acc = 0.0
    historial     = []
    sin_mejora    = 0          # contador para early stopping

    print(f"\n{'Época':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
    print("─" * 55)

    for epoca in range(1, args.epocas + 1):
        tr_loss, tr_acc = entrenar_epoca(modelo, train_loader, criterio, optimizador, device)
        va_loss, va_acc = evaluar(modelo, val_loader, criterio, device)
        scheduler.step()

        historial.append((tr_loss, tr_acc, va_loss, va_acc))

        if va_acc > mejor_val_acc:
            mejor_val_acc = va_acc
            sin_mejora    = 0
            torch.save(modelo.state_dict(), args.modelo)
            marca = " ✓"
        else:
            sin_mejora += 1
            restantes = args.patience - sin_mejora
            marca = f" (sin mejora {sin_mejora}/{args.patience})"

        print(f"{epoca:>6}  {tr_loss:>10.4f}  {tr_acc:>8.1%}  {va_loss:>8.4f}  {va_acc:>6.1%}{marca}")

        if sin_mejora >= args.patience:
            print(f"\n⏹  Early stopping en época {epoca} (patience={args.patience} alcanzado).")
            break

    print(f"\nMejor val acc : {mejor_val_acc:.1%}")
    print(f"Modelo guardado en: {args.modelo}")

    # Graficar curvas si matplotlib está disponible
    try:
        import matplotlib.pyplot as plt
        epocas_range = range(1, len(historial) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epocas_range, [h[0] for h in historial], label="Train")
        ax1.plot(epocas_range, [h[2] for h in historial], label="Val")
        ax1.set(title="Loss", xlabel="Época", ylabel="Loss")
        ax1.legend()
        ax2.plot(epocas_range, [h[1]*100 for h in historial], label="Train")
        ax2.plot(epocas_range, [h[3]*100 for h in historial], label="Val")
        ax2.set(title="Accuracy (%)", xlabel="Época", ylabel="Acc")
        ax2.legend()
        fig.tight_layout()
        plot_path = Path(args.modelo).with_suffix(".png")
        fig.savefig(plot_path, dpi=120)
        print(f"Curvas guardadas en: {plot_path}")
    except ImportError:
        pass


# ─────────────────────────────────────────
# Normalización para predict
# ─────────────────────────────────────────
def normalizar_para_predict(img: Image.Image, debug: bool = False) -> Image.Image:
    """
    Preprocesa una imagen de bola de bingo real para el modelo:
    1. Escala manteniendo proporciones para que quepa en IMG_W x IMG_H.
    2. Centra con padding negro (fondo de bolas de bingo) en ambos lados.
    """
    img = img.convert("RGB")
    w, h = img.size
    escala = min(IMG_W / w, IMG_H / h)
    nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
    scaled = img.resize((nw, nh), Image.LANCZOS)
    lienzo = Image.new("RGB", (IMG_W, IMG_H), color=(0, 0, 0))  # padding negro (fondo bolas)
    lienzo.paste(scaled, ((IMG_W - nw) // 2, (IMG_H - nh) // 2))
    if debug:
        print(f"  [preproceso] {w}x{h} -> {nw}x{nh} con padding negro a {IMG_W}x{IMG_H}")
    return lienzo


# ─────────────────────────────────────────
# Comando: predict
# ─────────────────────────────────────────
@torch.no_grad()
def cmd_predict(args):
    import numpy as np
    device = get_device()
    modelo = CNN_Numeros(num_clases=NUM_CLASES).to(device)
    modelo.load_state_dict(torch.load(args.modelo, map_location=device, weights_only=True))
    modelo.eval()

    img = Image.open(args.imagen).convert("RGB")
    w_orig, h_orig = img.size

    debug = getattr(args, "debug", False)
    img_norm = normalizar_para_predict(img, debug=debug)

    if debug:
        p = Path(args.imagen); debug_path = p.with_name(p.stem + "_preprocesada" + p.suffix)
        img_norm.save(debug_path)
        print(f"  [debug] Imagen preprocesada guardada en: {debug_path}")

    from torchvision import transforms as T
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    tensor = transform(img_norm).unsqueeze(0).to(device)

    logits = modelo(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    top5   = probs.topk(5)

    print(f"\nImagen   : {args.imagen}  ({w_orig}x{h_orig} px original)")
    print(f"Normalizada a {IMG_W}x{IMG_H} px con padding negro")
    print(f"\n{'Predicción':>12}  {'Prob':>6}")
    print("─" * 22)
    for prob, idx in zip(top5.values, top5.indices):
        numero = idx.item() + 1
        print(f"{numero:>12}  {prob.item():>5.1%}")


# ─────────────────────────────────────────
# Comando: eval
# ─────────────────────────────────────────
@torch.no_grad()
def cmd_eval(args):
    device = get_device()
    modelo = CNN_Numeros(num_clases=NUM_CLASES).to(device)
    modelo.load_state_dict(torch.load(args.modelo, map_location=device, weights_only=True))
    modelo.eval()

    dataset  = NumerosDataset(Path(args.datos), transform=get_transform(augment=False))
    loader   = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    criterio = nn.CrossEntropyLoss()

    todas_preds, todas_labels = [], []
    total_loss, total = 0.0, 0

    for imgs, etiquetas in loader:
        imgs, etiquetas = imgs.to(device), etiquetas.to(device)
        salidas = modelo(imgs)
        loss    = criterio(salidas, etiquetas)
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        todas_preds.extend(salidas.argmax(1).cpu().tolist())
        todas_labels.extend(etiquetas.cpu().tolist())

    acc_global = sum(p == l for p, l in zip(todas_preds, todas_labels)) / total

    print(f"\nDataset : {args.datos}  ({total} imágenes)")
    print(f"Loss    : {total_loss/total:.4f}")
    print(f"Acc     : {acc_global:.1%}")

    # Accuracy por clase
    from collections import defaultdict
    correctos_clase = defaultdict(int)
    total_clase     = defaultdict(int)
    errores         = []

    for pred, label in zip(todas_preds, todas_labels):
        total_clase[label] += 1
        if pred == label:
            correctos_clase[label] += 1
        else:
            errores.append((label + 1, pred + 1))

    clases_mal = [(lbl, c, total_clase[lbl]) for lbl, c in correctos_clase.items()
                  if c < total_clase[lbl]]

    if clases_mal:
        print(f"\nClases con errores ({len(clases_mal)}):")
        print(f"  {'Número':>7}  {'Correctas':>9}  {'Total':>6}  {'Acc':>6}")
        print("  " + "─" * 32)
        for lbl, corr, tot in sorted(clases_mal, key=lambda x: x[1]/x[2]):
            print(f"  {lbl+1:>7}  {corr:>9}  {tot:>6}  {corr/tot:>5.0%}")
    else:
        print("\n✓ Sin errores en ninguna clase.")

    if errores:
        print(f"\nEjemplos de errores (real → predicho):")
        for real, pred in errores[:10]:
            print(f"  {real} → {pred}")


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CNN para clasificar números 1-75 compuestos de imágenes de dígitos."
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    # ── train ────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Entrenar la CNN.")
    p_train.add_argument("--datos",   required=True,         help="Carpeta raíz con subcarpetas 1/ 2/ ... 75/ (ej: train/).")
    p_train.add_argument("--modelo",  default=MODELO_PATH,   help=f"Ruta para guardar el modelo (default: {MODELO_PATH}).")
    p_train.add_argument("--epocas",   type=int,   default=EPOCAS,    help=f"Número máximo de épocas (default: {EPOCAS}).")
    p_train.add_argument("--patience", type=int,   default=PATIENCE,  help=f"Early stopping: épocas sin mejora (default: {PATIENCE}). 0 = desactivado.")
    p_train.add_argument("--lr",       type=float, default=LR,        help=f"Learning rate (default: {LR}).")
    p_train.add_argument("--batch",    type=int,   default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE}).")

    # ── predict ──────────────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="Predecir el número en una imagen.")
    p_pred.add_argument("--imagen",  required=True, help="Ruta a la imagen a clasificar.")
    p_pred.add_argument("--modelo",  default=MODELO_PATH, help=f"Ruta al modelo guardado (default: {MODELO_PATH}).")
    p_pred.add_argument("--debug",   action="store_true",  help="Guarda la imagen preprocesada para verificar el pipeline.")

    # ── eval ─────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluar el modelo sobre una carpeta.")
    p_eval.add_argument("--datos",   required=True, help="Carpeta raíz con subcarpetas 1/ 2/ ... 75/ (ej: train/).")
    p_eval.add_argument("--modelo",  default=MODELO_PATH, help=f"Ruta al modelo guardado (default: {MODELO_PATH}).")
    p_eval.add_argument("--batch",   type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE}).")

    args = parser.parse_args()

    # Verificar PyTorch
    try:
        import torch  # noqa: F401
    except ImportError:
        print("Error: PyTorch no está instalado. Ejecuta:")
        print("  pip install torch torchvision")
        sys.exit(1)

    if args.comando == "train":
        cmd_train(args)
    elif args.comando == "predict":
        cmd_predict(args)
    elif args.comando == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()