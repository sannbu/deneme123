"""Sabit kameralı IR dizisinde etiketli objeleri silip arka plan üretir, sonucu kendi kendine puanlar (v2).

Her karede etiketli objelerin birleşik maskesi bölgelere ayrılır. Her bölge için aday dolgular üretilir:

  temporal_box     Bölgenin tamamının boş olduğu en yakın kareden ham kopya (bg_clean.py yöntemi).
  temporal_box_pb  Aynı kaynak, gradyan alanında birleştirme (Poisson): kenardaki parlaklık ve bulut
                   farkı yumuşakça giderilir, doku kaynaktan gelir.
  mosaic_pb        Her piksel kendi açıldığı en yakın kareden alınır. Yavaş objenin ön kenarı yakın
                   geçmişte, arka kenarı yakın gelecekte boştur; bütün kutunun açılmasını beklemek
                   gerekmez. Parçalar Poisson ile dikişsiz birleştirilir.
  mosaic_flow_pb   Mozaik ile aynı, kaynak kareler optik akışla mevcut kareye hizalanır (sürüklenen bulut).
  spatial          Yalnızca komşu piksellerden harmonik dolgu + sensör gürültüsü, zamansal kaynak yoksa yedek.

Hangi adayın kullanılacağına sahnenin kendisi karar verir: bölgenin yakınında objesiz, aynı biçimli
prob bölgeleri maskelenir ve her aday onları da doldurur. Proba objenin zaman içindeki doluluğu aynen
aktarılır, yani aday prob için de obje için gereken zaman uzaklığına gitmek zorunda kalır. Probun gerçek
pikselleriyle farkı, adayın o konumdaki beklenen hatasıdır. Hata, aynı bölgenin komşu kareyle farkına
(gürültü tabanı) veya yerel parlaklık aralığının %1.5'ine (görünürlük payı), hangisi büyükse ona bölünür:
1 kusursuz zamansal kopya düzeyi, 0.8'e kadar iyi, 2.5 ve üstü kötü.

Puan dört ölçütün en zayıfıdır: prob hatası, kalıntı (etiket çevresinde arka plana uymayan piksel oranı,
yani objenin ne kadar kapatıldığı), dikiş (bölge sınırındaki geçişin çevredeki doğal geçişe oranı) ve doku
(dolgunun ince doku enerjisinin çevresine oranı; çevrede olmayan yapı veya fazla pürüzsüzlük göze batar).
Seçilen dolgunun dokusu çevresinin 2.5 katından farklıysa sıradaki en iyi adaya geçilir.
Karedeki en kötü bölgenin puanına göre kare iyi, orta veya kotu klasörüne yazılır.

Ölçek: her obje için verilen ölçeklerden (varsayılan 1.0-1.4) kutunun hemen dışında ısı halesi
bırakmayan en küçüğü seçilir. Bit derinliği korunur (8/14/16), bölgeler dışındaki pikseller değişmez.

v2 eklemeleri (v1'in üstüne):
  Obje biçimli doluluk  (Deneysel, --footprint.) Bir pikselin bir karede açık olup olmadığına etiket kutusuyla
                        değil, kutunun içindeki objenin gerçek sıcak iziyle karar verilir; gökyüzünde bekleme kısalır.
                        Dokulu zeminde objenin zayıf parçalarını kaçırabildiği için varsayılan kapalıdır.
  Büyük bölge nakli     Puanı kötü çıkan bölge için bölge 2-3 kat büyütülür, bu pencerenin tamamen boş olduğu
                        ve kenarında mevcut kareyle en iyi uyuşan kare bulunur ve pencere Poisson ile yapıştırılır.
                        Yalnız v1 dolgusunda gözle görülür kusur (kalıntı veya dikiş) varsa denenir ve bağışçının
                        pencere kenarı mevcut kareyle gürültü düzeyinde uyuşmalıdır; sürüklenen bulutta bu koşul
                        sağlanmaz ve nakil yapılmaz (orada başka anın bulutu tutarsız görünür). Sonuç o anın gerçek
                        arka planı değil, tutarlı gerçek bir sahne parçasıdır; en fazla "orta" sayılır, raporda
                        "transplant" yazar.
  Güven haritası        Her kare için guven/<ad>.png: 255 dokunulmamış, 200 iyi dolgu, 128 orta dolgu, 0 kötü
                        bölge. Tek bir kötü obje yüzünden kare çöpe gitmez; eğitimde 0 olan yer yok sayılabilir.

Kurulum:
    pip install numpy opencv-python scipy

Örnek:
    python bg_sequence_v2.py --project "C:\\proje" --out "C:\\cikti"
    python bg_sequence_v2.py --frames tif --labels etiketler --out cikti --preview-every 20
"""
import argparse
import csv
import json
import math
import re
import shutil
import struct
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

TIFF_SUFFIXES = (".tif", ".tiff")
BITS_PER_SAMPLE_TAG = 258
FRAMES_SUBDIR = Path("2_Etiketlenecek_Veri") / "tif"
COCO_SUBDIR = Path("3_Etiketlenmis_Veri_Ciktisi")
CATEGORIES = ("iyi", "orta", "kotu")
ORIGINAL_SUBDIR = "silinmemis"
PREVIEW_SUBDIR = "onizleme"
CONFIDENCE_SUBDIR = "guven"
CONFIDENCE_VALUES = {"iyi": 200, "orta": 128, "kotu": 0}
METHODS = ("temporal_box", "temporal_box_pb", "mosaic_pb", "mosaic_flow_pb", "interp_pb", "interp_flow_pb",
           "spatial")
SHORT_NAMES = {"transplant": "nakil", "temporal_box": "kutu", "temporal_box_pb": "kutu+pb", "mosaic_pb": "mozaik",
               "mosaic_flow_pb": "mozaik+akis", "interp_pb": "aradeger", "interp_flow_pb": "aradeger+akis",
               "spatial": "uzamsal"}
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))
CONTEXT_MIN = 12
SEARCH_WINDOWS = (24, 96, 300)
PROBE_STEPS = (1.3, 2.0, 2.6, 3.4)
STRUCTURE_MATCH = 1.6
STRUCTURE_RATIO_CAP = 8.0
PROBE_PAD = 3
NOISE_FLOOR_FRAMES = 3
VISIBLE_FRACTION = 0.015
HALO_REACH = 0.8
HALO_MIN_REACH = 12
REFERENCE_OFFSETS = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987)
RESIDUAL_Z = 6.0
COVERAGE_SCALE = 1.15
LEFTOVER_BAD = 0.10
ERROR_GOOD, ERROR_BAD = 0.8, 2.5
SEAM_GOOD, SEAM_BAD = 1.5, 3.5
NO_PROBE_SCORE_CAP = 0.6
TEXTURE_VETO = 2.5
TEXTURE_GOOD, TEXTURE_BAD = 1.6, 3.0
CACHE_SIZE = 48
FLOW_OBJECT_CLEARANCE = 16
FLOW_MIN_PIXELS = 200
CACHE_BYTES = 300 * 2 ** 20

FOOTPRINT_Z = 4.0
FOOTPRINT_RING = 4
FOOTPRINT_DILATE = 5
TRANSPLANT_FACTORS = (2.0, 3.0)
TRANSPLANT_RING = 3
TRANSPLANT_DONORS = 8
TRANSPLANT_CAP = 0.8
TRANSPLANT_MATCH = 2.5
TRANSPLANT_LEFTOVER = 0.05

_thread_state = threading.local()
# Obje izleri süreç içinde kare başına bir kez hesaplanır, ardışık işlerde yeniden kullanılır.
_FOOTPRINTS = {}


# ---------------------------------------------------------------- okuma / yazma

def frame_number(path):
    match = re.search(r"(\d+)(?!.*\d)", Path(path).stem)
    if match is None:
        raise ValueError(f"Dosya adında kare numarası yok: {path}")
    return int(match.group(1))


def list_frames(frames_dir):
    paths = [p for p in Path(frames_dir).iterdir() if p.suffix.lower() in TIFF_SUFFIXES]
    if not paths:
        raise FileNotFoundError(f"TIFF bulunamadı: {frames_dir}")
    return sorted(paths, key=frame_number)


def tiff_bits(data):
    """TIFF başlığındaki piksel başına bit sayısı, BigTIFF için None."""
    order = "<" if bytes(data[:2]) == b"II" else ">"
    if struct.unpack_from(order + "H", data, 2)[0] != 42:
        return None
    offset = struct.unpack_from(order + "I", data, 4)[0]
    for entry in range(struct.unpack_from(order + "H", data, offset)[0]):
        position = offset + 2 + 12 * entry
        tag, _, count = struct.unpack_from(order + "HHI", data, position)
        if tag == BITS_PER_SAMPLE_TAG:
            value = position + 8
            if count > 2:
                value = struct.unpack_from(order + "I", data, value)[0]
            return struct.unpack_from(order + "H", data, value)[0]
    return 1


# cv2.imread/imwrite Windows'ta ASCII dışı yollarda (Masaüstü gibi) çalışmaz.
def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    bits = tiff_bits(data) if Path(path).suffix.lower() in TIFF_SUFFIXES else None
    # OpenCV paketli 14 bit TIFF'i hata vermeden yanlış değerlerle okur.
    if bits is not None and bits % 8:
        raise IOError(f"{bits} bit paketli TIFF desteklenmiyor, önce 16 bit kapsayıcıya çevrilmeli: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Okunamadı: {path}")
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise IOError(f"Yazılamadı: {path}")
    # Yazılacak dosya bit derinliği ve piksel değerleriyle birebir aynı olmalı.
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != image.dtype or not np.array_equal(decoded, image):
        raise IOError(f"Yazılan veri girdiyle aynı değil: {path}")
    encoded.tofile(str(path))


def label_box(x, y, w, h, width, height):
    """Ham etiket kutusu (x1, y1, x2, y2), görüntüye kırpılmış; görüntü dışındaysa None."""
    x1, y1 = max(0.0, float(x)), max(0.0, float(y))
    x2, y2 = min(float(width), float(x) + float(w)), min(float(height), float(y) + float(h))
    return (x1, y1, x2, y2) if x2 - x1 >= 1 and y2 - y1 >= 1 else None


def load_yolo_labels(labels_dir, frames, width, height):
    """Kare başına `sınıf cx cy w h` satırlı, 0-1 normalize YOLO .txt dosyaları."""
    boxes = []
    for frame in frames:
        path = Path(labels_dir) / f"{frame.stem}.txt"
        frame_boxes = []
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    cx, cy, w, h = map(float, line.split()[1:5])
                    box = label_box((cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height, width, height)
                    if box:
                        frame_boxes.append(box)
        boxes.append(frame_boxes)
    return boxes


def load_csv_labels(csv_path, frames, width, height):
    """`frame,x,y,w,h` başlıklı CSV; x,y sol üst köşe, tüm değerler piksel."""
    position = {frame_number(frame): i for i, frame in enumerate(frames)}
    boxes = [[] for _ in frames]
    with open(csv_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            index = position.get(int(row["frame"]))
            box = label_box(row["x"], row["y"], row["w"], row["h"], width, height)
            if index is not None and box:
                boxes[index].append(box)
    return boxes


def load_coco_labels(images_path, annotations_path, frames, width, height):
    """`bbox` değeri piksel [x, y, w, h] olan COCO dosyaları, kareler uzantısız adla eşleşir.

    COCO'da kaydı olmayan karenin içeriği bilinmediği için kutu listesi None olur.
    """
    images = json.loads(Path(images_path).read_text(encoding="utf-8"))["images"]
    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))["annotations"]
    position = {frame.stem: i for i, frame in enumerate(frames)}
    boxes = [None] * len(frames)
    image_position = {}
    for image in images:
        index = position.get(Path(image["file_name"]).stem)
        if index is not None:
            image_position[image["id"]] = index
            boxes[index] = []
    for annotation in annotations:
        index = image_position.get(annotation["image_id"])
        box = label_box(*annotation["bbox"], width, height) if index is not None else None
        if box:
            boxes[index].append(box)
    return boxes


def sequence_name(frame):
    stem = Path(frame).stem
    return stem[:-6] if stem[-6:].isdigit() else stem.rstrip("0123456789")


def split_sequences(frames, boxes):
    """Kareleri dosya adı önekine göre dizilere ayırır, kutu listesi None olan kareleri atar."""
    sequences = {}
    for frame, frame_boxes in zip(frames, boxes):
        if frame_boxes is not None:
            sequence_frames, sequence_boxes = sequences.setdefault(sequence_name(frame), ([], []))
            sequence_frames.append(frame)
            sequence_boxes.append(frame_boxes)
    return list(sequences.values())


# ---------------------------------------------------------------- geometri

def scale_box(box, scale, pad, width, height):
    """Etiketi merkezden ölçekler ve tam sayı piksel kutusuna çevirir (x0, y0, x1, y1)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    # Yuvarlama hatası kutuyu 1 piksel büyütmesin.
    return (max(0, math.floor(round(cx - half_w, 2)) - pad), max(0, math.floor(round(cy - half_h, 2)) - pad),
            min(width, math.ceil(round(cx + half_w, 2)) + pad), min(height, math.ceil(round(cy + half_h, 2)) + pad))


def plane(image, channel=None):
    """Görüntünün float32 düzlemi: tek kanal veya renkli görüntüde istenen kanal / gri ortalama."""
    if image.ndim == 2:
        return image.astype(np.float32)
    if channel is None:
        return image[..., :3].astype(np.float32).mean(axis=2)
    return image[..., channel].astype(np.float32)


def pair_noise(image):
    """Komşu piksel farkından gürültü standart sapması (sahne yapısına dayanıklı)."""
    difference = np.abs(np.diff(image, axis=1))
    return max(float(np.median(difference)) * 1.4826 / math.sqrt(2), 1e-3)


def robust_stats(values):
    median = float(np.median(values))
    return median, 1.4826 * float(np.median(np.abs(values - median)))


def box_distance(shape, box):
    """Her pikselin dikdörtgen kutuya satranç tahtası uzaklığı (kutunun içi 0)."""
    x0, y0, x1, y1 = box
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    dx = np.maximum(np.maximum(x0 - xx, xx - (x1 - 1)), 0)
    dy = np.maximum(np.maximum(y0 - yy, yy - (y1 - 1)), 0)
    return np.maximum(dx, dy)


def detrend(values, reference):
    """Referans piksellere düzlem oturtup tüm kırpıntıdan çıkarır: gökyüzü eğimi ve bulut seviyesi gider."""
    ys, xs = np.nonzero(reference)
    if len(ys) < 12:
        return values - (float(np.median(values[reference])) if reference.any() else float(values.mean()))
    design = np.column_stack([np.ones(len(ys)), xs, ys]).astype(np.float64)
    coefficients = np.linalg.lstsq(design, values[ys, xs].astype(np.float64), rcond=None)[0]
    yy, xx = np.mgrid[:values.shape[0], :values.shape[1]]
    return values - (coefficients[0] + coefficients[1] * xx + coefficients[2] * yy).astype(np.float32)


def ring_profile(residual, distance, usable, box, near, far, polarity=None):
    """1 piksellik halkalarda hale sinyali ve uzak halkalara göre oynama.

    Halka dört kenara bölünür. Hale objenin her yanında aynı işaretle görülür, bulut kenarı gibi sahne
    yapısı ise genelde bir iki yanda; bu yüzden sinyal, dört kenar medyanı aynı işaretliyse en küçüğünün
    mutlak değeridir, değilse sıfırdır. polarity verilirse (sıcak obje +1) yalnız o işaret sayılır: bulut
    içindeki karanlık bir boşluk sıcak objenin halesi sanılmaz.
    """
    x0, y0, x1, y1 = box
    yy, xx = np.mgrid[:residual.shape[0], :residual.shape[1]]
    sides = (yy < y0, yy >= y1, (xx < x0) & (yy >= y0) & (yy < y1), (xx >= x1) & (yy >= y0) & (yy < y1))
    signals = {}
    for d in list(near) + list(far):
        ring = usable & (distance == d)
        medians = [float(np.median(residual[ring & side])) for side in sides if (ring & side).sum() >= 2]
        if len(medians) >= 2:
            if polarity:
                signed = [m * polarity for m in medians]
                signals[d] = min(signed) if all(v > 0 for v in signed) else 0.0
                continue
            same_sign = all(m > 0 for m in medians) or all(m < 0 for m in medians)
            signals[d] = min(abs(m) for m in medians) if same_sign else 0.0
    far_values = [signals[d] for d in far if d in signals]
    spread = float(np.std(far_values) + np.mean(far_values)) if len(far_values) >= 3 else 0.0
    return signals, spread


# ---------------------------------------------------------------- zamansal doluluk

def trail_valid(occupied, margin):
    """t anında geçerli: piksel t-margin..t aralığında hiç dolu olmamış (ısıl iz sönmüş)."""
    counts = np.cumsum(occupied, axis=0, dtype=np.int32)
    before = np.zeros_like(counts)
    before[margin + 1:] = counts[:-margin - 1]
    return (counts - before) == 0


def mosaic_assignment(valid, mask, center, first, offsets, max_sources, min_gain):
    """Her maske pikseline açıldığı en yakın kareyi atar; az kaynak, az dikiş için küçük kazançlar atlanır."""
    ids = np.full(mask.shape, -1, np.int32)
    remaining = mask.copy()
    left = int(remaining.sum())
    order = []
    for offset in offsets:
        for t in (center - offset, center + offset):
            index = t - first
            if not 0 <= index < len(valid):
                continue
            gain = valid[index] & remaining
            count = int(gain.sum())
            if count == 0 or (count < left and (count < min_gain or len(order) == max_sources - 1)):
                continue
            ids[gain] = len(order)
            order.append(t)
            remaining &= ~gain
            left -= count
            if left == 0:
                return order, ids
    return None


def directional_assignment(valid, mask, center, first, offsets, sign, max_sources, min_gain):
    """Yalnız geçmişe (sign=-1) veya geleceğe (+1) bakan atama; kapsanamayan pikseller -1 kalır."""
    ids = np.full(mask.shape, -1, np.int32)
    remaining = mask.copy()
    left = int(remaining.sum())
    order = []
    for offset in offsets:
        index = center + sign * offset - first
        if not 0 <= index < len(valid):
            continue
        gain = valid[index] & remaining
        count = int(gain.sum())
        if count == 0 or (count < left and (count < min_gain or len(order) == max_sources - 1)):
            continue
        ids[gain] = len(order)
        order.append(center + sign * offset)
        remaining &= ~gain
        left -= count
        if left == 0:
            break
    return (order, ids) if order else None


# ---------------------------------------------------------------- gradyan alanında birleştirme

class PoissonSystem:
    """Maske içindeki pikseller için Poisson denklemi; matris maskenin biçimine bağlıdır ve bir kez
    çözümlenir, aynı biçimli tüm prob ve kanallarda yeniden kullanılır.

    Maske dışındaki komşular sınır koşuludur (Dirichlet), görüntü dışı komşular yok sayılır (Neumann).
    """

    def __init__(self, mask):
        height, width = mask.shape
        self.ys, self.xs = np.nonzero(mask)
        n = len(self.ys)
        self.n = n
        index = np.full(mask.shape, -1, np.int64)
        index[self.ys, self.xs] = np.arange(n)
        diagonal = np.zeros(n)
        rows, cols, data = [np.arange(n)], [np.arange(n)], []
        self.links = []
        for dy, dx in DIRECTIONS:
            qy, qx = self.ys + dy, self.xs + dx
            inside = (qy >= 0) & (qy < height) & (qx >= 0) & (qx < width)
            diagonal += inside
            p = np.nonzero(inside)[0]
            qy, qx = qy[inside], qx[inside]
            q = index[qy, qx]
            self.links.append((p, qy, qx, q))
            linked = q >= 0
            rows.append(p[linked])
            cols.append(q[linked])
            data.append(-np.ones(int(linked.sum())))
        matrix = sparse.csc_matrix((np.concatenate([diagonal] + data), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(n, n))
        self.solve = factorized(matrix)

    def blend(self, target, sources, ids, valids):
        """target (h, w) sınırı, sources (k, h, w) yönlendirme gradyanı, ids (n,) her pikselin kaynağı,
        valids (k, h, w) kaynağın o pikselde objesiz olup olmadığı. Maske içi değerleri döndürür."""
        b = np.zeros(self.n)
        ys, xs = self.ys, self.xs
        for p, qy, qx, q in self.links:
            own = ids[p]
            py, px = ys[p], xs[p]
            first = sources[own, py, px] - sources[own, qy, qx]
            # Kaynak hem p hem q'da objesiz olmalı; akışla kaydırılan kaynakta obje örneklenmiş olabilir.
            first_ok = valids[own, qy, qx] & valids[own, py, px]
            guidance = np.where(first_ok, first, 0.0)
            inner = q >= 0
            if inner.any():
                other = ids[q[inner]]
                second = sources[other, py[inner], px[inner]] - sources[other, qy[inner], qx[inner]]
                second_ok = valids[other, py[inner], px[inner]] & valids[other, qy[inner], qx[inner]]
                f, f_ok = first[inner], first_ok[inner]
                guidance[inner] = np.where(f_ok & second_ok, (f + second) / 2,
                                           np.where(f_ok, f, np.where(second_ok, second, 0.0)))
            outer = ~inner
            guidance[outer] += target[qy[outer], qx[outer]]
            b[p] += guidance
        return self.solve(b)


def texture_ratio(target, mask, values, ys, xs):
    """Dolgunun ince doku enerjisinin çevresine oranı (1 uyumlu); ters yönde de (dolgu çok pürüzsüz) büyür."""
    filled = target.astype(np.float32).copy()
    filled[ys, xs] = values
    detail = filled - cv2.GaussianBlur(filled, (0, 0), 1.5)
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~mask
    inner, outer = float(np.std(detail[mask])), float(np.std(detail[ring])) if ring.any() else 0.0
    ratio = inner / max(outer, 1e-6)
    return max(ratio, 1 / max(ratio, 1e-6))


def structure_level(values, ring):
    """Halkada düzlemden sapmanın gürültü düşülmüş standart sapması: çevrenin ne kadar yapılı olduğu."""
    if ring.sum() < 12:
        return None
    spread = float(np.std(detrend(values, ring)[ring]))
    return math.sqrt(max(spread ** 2 - pair_noise(values) ** 2, 0.0))


def transfer_error(error, ratio):
    """Prob hatasını bölgeye taşır. Bölge probdan daha yapılıysa (ratio > 1) hatanın gürültüyü aşan kısmı
    oranla büyür: bulut değişimi ve uzamsal tahmin hatası yapıyla artar, kusursuz kopyanın gürültüsü artmaz."""
    if not np.isfinite(error) or ratio <= 1 or error <= 1:
        return error
    return 1 + (error - 1) * min(ratio, STRUCTURE_RATIO_CAP)


def spatial_fill(system, target):
    """Yalnız komşu piksellerden harmonik (en pürüzsüz) dolgu. Telea büyük alanda X biçimli iz bırakıyordu.
    Pürüzsüz dolgu çevresinden ayırt edilmesin diye bağlamdaki sensör gürültüsü eklenir."""
    zeros = np.zeros((1,) + target.shape, np.float32)
    everywhere = np.ones((1,) + target.shape, bool)
    values = system.blend(target, zeros, np.zeros(system.n, np.int64), everywhere)
    rng = np.random.default_rng(system.n)
    return values + rng.normal(0, pair_noise(target), system.n)


def dis_flow(target8, source8):
    if not hasattr(_thread_state, "dis"):
        _thread_state.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _thread_state.dis.calc(target8, source8, None)


def background_flow(target, source, hole):
    """target->source optik akış. Obje bölgesinde akış güvenilmez, çevredeki akışın medyanıyla doldurulur."""
    reference = target[~hole] if (~hole).any() else target.ravel()
    low, high = np.percentile(reference, (1, 99))
    scale = 255.0 / max(high - low, 1e-6)
    to8 = lambda image: np.clip((image - low) * scale, 0, 255).astype(np.uint8)
    try:
        flow = dis_flow(to8(target), to8(source))
    except cv2.error:
        return np.zeros(target.shape + (2,), np.float32)
    ring = cv2.dilate(hole.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    ring &= ~hole
    if ring.any():
        flow[hole] = np.median(flow[ring], axis=0)
    else:
        flow[hole] = 0
    return flow


class FlowCache:
    """Mevcut kare ile her kaynak kare arasındaki arka plan kayması; bölgeyi ve problarını kapsayan pencerede
    optik akıştan, objelerden uzak piksellerin medyanı olarak bir kez hesaplanır ve kırpıntıya sabit uygulanır."""

    def __init__(self, sequence, store, center, gray, crops):
        self.sequence, self.store, self.center = sequence, store, center
        self.rect = (min(c[0] for c in crops), max(c[1] for c in crops), min(c[2] for c in crops), max(c[3] for c in crops))
        y0, y1, x0, x1 = self.rect
        self.target = gray[y0:y1, x0:x1]
        low, high = np.percentile(self.target, (1, 99))
        self.low, self.scale = low, 255.0 / max(high - low, 1e-6)
        self.raw = {}
        self.shift = {}

    def to8(self, image):
        return np.clip((image - self.low) * self.scale, 0, 255).astype(np.uint8)

    def get(self, t, crop, mask):
        if t not in self.raw:
            try:
                self.raw[t] = dis_flow(self.to8(self.target), self.to8(self.store.crop(t, self.rect)))
            except cv2.error:
                self.raw[t] = np.zeros(self.target.shape + (2,), np.float32)
        if t not in self.shift:
            # Arka plan kayması (bulut sürüklenmesi, kamera titremesi) objelerden uzak piksellerin medyanıdır:
            # akış hesabı hareketli objenin hareketini çevresine yaydığı için obje yakını kullanılmaz.
            objects = self.sequence.occupancy(t, t, self.rect)[0] | self.sequence.occupancy(self.center, self.center, self.rect)[0]
            size = 2 * FLOW_OBJECT_CLEARANCE + 1
            near = cv2.dilate(objects.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
            background = self.raw[t][~near]
            self.shift[t] = np.median(background, axis=0) if len(background) >= FLOW_MIN_PIXELS else np.zeros(2, np.float32)
        return np.broadcast_to(self.shift[t].astype(np.float32), (crop[1] - crop[0], crop[3] - crop[2], 2)).copy()


def warp(image, flow, interpolation=cv2.INTER_LINEAR):
    height, width = image.shape
    gx, gy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(image, gx + flow[..., 0], gy + flow[..., 1], interpolation, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- dizi işleme

class FrameStore:
    def __init__(self, paths, size=CACHE_SIZE):
        self.paths, self.size, self.cache = paths, size, OrderedDict()

    def raw(self, t):
        if t in self.cache:
            self.cache.move_to_end(t)
        else:
            self.cache[t] = read_image(self.paths[t])
            if len(self.cache) > self.size:
                self.cache.popitem(last=False)
        return self.cache[t]

    def crop(self, t, rect, channel=None):
        y0, y1, x0, x1 = rect
        return plane(self.raw(t)[y0:y1, x0:x1], channel)


class RegionPlan:
    """Bir bölge (veya prob) için kaynak seçimleri; her yöntem bunu kullanarak dolgu üretir."""

    def __init__(self, center):
        self.center = center
        self.box_source = None
        self.mosaic = None
        self.interp = None
        self.valids = {}
        self.flows = {}


class Sequence:
    def __init__(self, frames, boxes, args):
        self.frames, self.boxes, self.args = frames, boxes, args
        first = read_image(frames[0])
        self.height, self.width = first.shape[:2]
        self.dtype = first.dtype
        self.methods = [m for m in METHODS if m in args.methods]
        self.store = None
        self.scales = sorted(args.scales)
        # Zamansal geçerlilik etiketten geniş tutulur, hale ve ısıl iz kaynak olarak kullanılmasın.
        self.occupancy_boxes = [
            [scale_box(box, args.occupancy_scale, args.occupancy_pad, self.width, self.height) for box in frame_boxes]
            for frame_boxes in boxes]

    # ---- doluluk

    def occupancy(self, first, last, rect):
        y0, y1, x0, x1 = rect
        occupied = np.zeros((last - first + 1, y1 - y0, x1 - x0), bool)
        for t in range(first, last + 1):
            footprints = None
            for k, (bx0, by0, bx1, by1) in enumerate(self.occupancy_boxes[t]):
                ix0, ix1, iy0, iy1 = max(bx0, x0), min(bx1, x1), max(by0, y0), min(by1, y1)
                if ix0 < ix1 and iy0 < iy1:
                    if self.args.footprint and self.store is not None:
                        footprints = footprints or self.footprints(t)
                        occupied[t - first, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] |= \
                            footprints[k][iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
                    else:
                        occupied[t - first, iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = True
        return occupied

    def footprints(self, t):
        key = str(self.frames[t])
        if key not in _FOOTPRINTS:
            _FOOTPRINTS[key] = self.compute_footprints(t)
        return _FOOTPRINTS[key]

    def compute_footprints(self, t):
        """Kutu içinde objenin gerçek izi: kutunun dışındaki halkaya oturtulan düzlemden belirgin sapan pikseller,
        hale payı kadar genişletilmiş. Belirgin iz yoksa güvenli tarafta kalınır ve tüm kutu dolu sayılır."""
        gray = plane(self.store.raw(t))
        noise = pair_noise(gray)
        boxes = self.occupancy_boxes[t]
        masks = []
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            rx0, ry0 = max(0, x0 - FOOTPRINT_RING), max(0, y0 - FOOTPRINT_RING)
            rx1, ry1 = min(self.width, x1 + FOOTPRINT_RING), min(self.height, y1 + FOOTPRINT_RING)
            window = gray[ry0:ry1, rx0:rx1]
            inner = np.zeros(window.shape, bool)
            inner[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = True
            reference = ~inner
            for j, (ox0, oy0, ox1, oy1) in enumerate(boxes):
                if j != k:
                    reference[max(0, oy0 - ry0):max(0, oy1 - ry0), max(0, ox0 - rx0):max(0, ox1 - rx0)] = False
            full = np.ones((y1 - y0, x1 - x0), bool)
            if reference.sum() < 12:
                masks.append(full)
                continue
            residual = detrend(window, reference)
            _, spread = robust_stats(residual[reference])
            hot = (np.abs(residual) > FOOTPRINT_Z * (spread + noise)) & inner
            if hot.sum() < 4:
                masks.append(full)
                continue
            size = 2 * FOOTPRINT_DILATE + 1
            grown = cv2.dilate(hot.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
            masks.append(grown[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0])
        return masks

    def frame_occupancy(self, t):
        occupied = np.zeros((self.height, self.width), bool)
        for x0, y0, x1, y1 in self.occupancy_boxes[t]:
            occupied[y0:y1, x0:x1] = True
        return occupied

    # ---- geometri

    def context_crop(self, rect):
        y0, y1, x0, x1 = rect
        margin = max(CONTEXT_MIN, int(0.6 * max(y1 - y0, x1 - x0)))
        return (max(0, y0 - margin), min(self.height, y1 + margin), max(0, x0 - margin), min(self.width, x1 + margin))

    def temporal_reference(self, store, center, rect):
        """Her piksel için objelerin uzakta olduğu en yakın karedeki değer (1, 2, 3, 5, 8... kare ötede).

        Mevcut kareden çıkarıldığında ağaç, zemin gibi sabit dokular tamamen gider, geriye objeye ait hale
        ve yavaş sahne değişimi kalır.
        """
        offsets = [o for o in REFERENCE_OFFSETS if o <= self.args.search]
        first = max(0, center - offsets[-1] - self.args.margin)
        last = min(len(self.frames) - 1, center + offsets[-1])
        valid = trail_valid(self.occupancy(first, last, rect), self.args.margin)
        reference = np.zeros(valid.shape[1:], np.float32)
        done = np.zeros(valid.shape[1:], bool)
        for offset in offsets:
            for t in (center - offset, center + offset):
                if first <= t <= last:
                    take = valid[t - first] & ~done
                    if take.any():
                        reference[take] = store.crop(t, rect)[take]
                        done |= take
            if done.all():
                break
        return reference, done

    def object_polarity(self, gray, box, noise):
        """Objenin arka plana göre işareti: sıcak +1, soğuk -1, ayırt edilemiyorsa None."""
        x0, y0, x1, y1 = scale_box(box, 1.0, 0, self.width, self.height)
        inner = gray[y0:y1, x0:x1]
        oy0, oy1, ox0, ox1 = max(0, y0 - 6), min(self.height, y1 + 6), max(0, x0 - 6), min(self.width, x1 + 6)
        ring = np.ones((oy1 - oy0, ox1 - ox0), bool)
        ring[max(0, y0 - 2 - oy0):y1 + 2 - oy0, max(0, x0 - 2 - ox0):x1 + 2 - ox0] = False
        if inner.size == 0 or not ring.any():
            return None
        background = float(np.median(gray[oy0:oy1, ox0:ox1][ring]))
        hot, cold = float(np.percentile(inner, 95)) - background, background - float(np.percentile(inner, 5))
        if max(hot, cold) < 3 * noise:
            return None
        return 1 if hot >= cold else -1

    def halo_pad(self, store, center, gray, box, others, noise, polarity=None):
        """Etiketin dışında ısı halesinin kaç piksel sürdüğü.

        Etiketten dışa doğru 1 piksellik halkalara, bir arka plan modelinden farkta bakılır. İki model var:
        zamansal referans (objelerin uzakta olduğu en yakın kare; ağaç, zemin gibi sabit doku farkta yok olur)
        ve yalnız mevcut kareye oturtulan düzlem. Yavaş objede referans çok eski karelerden gelir, bulut o arada
        değiştiği için uzak halkalar oynar ve hale erken "bitmiş" görünür; bu yüzden uzak halkalarda daha az
        artık bırakan model kullanılır. Halka sinyali uzak halkaların oynamasının altına iki halka üst üste
        indiğinde hale bitmiş sayılır.
        """
        x0, y0, x1, y1 = scale_box(box, 1.0, 0, self.width, self.height)
        reach = max(HALO_MIN_REACH, int(HALO_REACH * max(x1 - x0, y1 - y0)))
        far = range(reach + 2, reach + 11)
        cx0, cy0 = max(0, x0 - far[-1]), max(0, y0 - far[-1])
        cx1, cy1 = min(self.width, x1 + far[-1]), min(self.height, y1 + far[-1])
        window = gray[cy0:cy1, cx0:cx1]
        distance = box_distance(window.shape, (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0))
        free = np.ones(window.shape, bool)
        for ox0, oy0, ox1, oy1 in others:
            free[max(0, oy0 - cy0 - PROBE_PAD):max(0, oy1 - cy0 + PROBE_PAD),
                 max(0, ox0 - cx0 - PROBE_PAD):max(0, ox1 - cx0 + PROBE_PAD)] = False
        models = [(window, free, noise)]
        reference, known = self.temporal_reference(store, center, (cy0, cy1, cx0, cx1))
        if known.mean() > 0.5:
            models.append((window - reference, free & known, noise * math.sqrt(2)))
        best = None
        for values, model_usable, model_noise in models:
            far_ring = model_usable & (distance >= far[0])
            model_residual = detrend(values, far_ring)
            roughness = robust_stats(model_residual[far_ring])[1] if far_ring.sum() >= 12 else np.inf
            if best is None or roughness < best[0]:
                best = (roughness, model_residual, model_usable, model_noise)
        _, residual, usable, noise = best
        local_box = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)
        signals, spread = ring_profile(residual, distance, usable, local_box, range(1, reach + 2), far, polarity)
        for d in range(1, reach + 1):
            calm = all(signals.get(e, 0.0) < self.halo_threshold(spread, noise, usable & (distance == e))
                       for e in (d, d + 1))
            if calm:
                return d
        return reach + 1

    @staticmethod
    def halo_threshold(spread, noise, ring):
        return max(3 * spread, 3 * noise / math.sqrt(max(int(ring.sum()), 1)), 0.5 * noise)

    def final_box(self, box, pad):
        """Haleyi kapsayan en küçük ölçek; hiçbiri yetmezse ölçülen hale payı kullanılır."""
        x0, y0, x1, y1 = scale_box(box, 1.0, 0, self.width, self.height)
        need = (max(0, x0 - pad), max(0, y0 - pad), min(self.width, x1 + pad), min(self.height, y1 + pad))
        for scale in self.scales:
            candidate = scale_box(box, scale, 0, self.width, self.height)
            if candidate[0] <= need[0] and candidate[1] <= need[1] and candidate[2] >= need[2] and candidate[3] >= need[3]:
                return candidate, scale
        return need, round((need[2] - need[0]) / max(x1 - x0, 1), 2)

    # ---- kaynak planı

    def make_plan(self, center, crop, mask, transplant=None):
        """Kutu ve mozaik kaynaklarını yakından uzağa, gerekirse arama penceresini büyüterek bulur.

        transplant verilirse (prob için) bu kırpıntının doluluğu da eklenir: prob, objenin zaman
        içindeki doluluğunu taşır.
        """
        args = self.args
        cy0, cy1, cx0, cx1 = crop
        ys, xs = np.nonzero(mask)
        ry0, ry1 = max(0, ys.min() - 1), min(mask.shape[0], ys.max() + 2)
        rx0, rx1 = max(0, xs.min() - 1), min(mask.shape[1], xs.max() + 2)
        local = (ry0, ry1, rx0, rx1)
        region_mask = mask[ry0:ry1, rx0:rx1]
        plan = RegionPlan(center)
        plan.local = local
        min_gain = max(16, int(0.04 * region_mask.sum()))
        for window in tuple(w for w in SEARCH_WINDOWS if w < args.search) + (args.search,):
            first = max(0, center - window - args.margin)
            last = min(len(self.frames) - 1, center + window)
            rect = (cy0 + ry0, cy0 + ry1, cx0 + rx0, cx0 + rx1)
            occupied = self.occupancy(first, last, rect)
            if transplant is not None:
                ty0, _, tx0, _ = transplant
                occupied |= self.occupancy(first, last, (ty0 + ry0, ty0 + ry1, tx0 + rx0, tx0 + rx1))
            valid = trail_valid(occupied, args.margin)
            offsets = range(args.min_offset, window + 1)
            if plan.box_source is None:
                complete = (valid | ~region_mask).all(axis=(1, 2))
                for offset in offsets:
                    for t in (center - offset, center + offset):
                        if 0 <= t - first < len(complete) and complete[t - first]:
                            plan.box_source = t
                            break
                    if plan.box_source is not None:
                        break
            if plan.mosaic is None and {"mosaic_pb", "mosaic_flow_pb"} & set(self.methods):
                found = mosaic_assignment(valid, region_mask, center, first, offsets, args.max_sources, min_gain)
                if found:
                    plan.mosaic = found
            last_window = window == args.search
            if {"interp_pb", "interp_flow_pb"} & set(self.methods) and (plan.interp is None or last_window):
                # Ara değer için maske 1 piksel genişletilir: sınırdaki gradyan da aynı kaynaktan gelsin.
                ring_mask = cv2.dilate(region_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                past = directional_assignment(valid, ring_mask, center, first, offsets, -1, args.max_sources, min_gain)
                future = directional_assignment(valid, ring_mask, center, first, offsets, 1, args.max_sources, min_gain)
                covered = np.zeros(ring_mask.shape, bool)
                for side in (past, future):
                    if side:
                        covered |= side[1] >= 0
                complete = all(side and not ((side[1] < 0) & ring_mask).any() for side in (past, future))
                if (complete or last_window) and not (region_mask & ~covered).any():
                    plan.interp = (past or ([], np.full(ring_mask.shape, -1, np.int32)),
                                   future or ([], np.full(ring_mask.shape, -1, np.int32)), ring_mask)
            used = ([plan.box_source] if plan.box_source is not None else []) + (plan.mosaic[0] if plan.mosaic else [])
            if plan.interp is not None:
                used += plan.interp[0][0] + plan.interp[1][0]
            for t in used:
                if t not in plan.valids:
                    full = np.ones(mask.shape, bool)
                    full[ry0:ry1, rx0:rx1] = valid[t - first]
                    plan.valids[t] = full
            box_done = plan.box_source is not None or not {"temporal_box", "temporal_box_pb"} & set(self.methods)
            mosaic_done = plan.mosaic is not None or not {"mosaic_pb", "mosaic_flow_pb"} & set(self.methods)
            interp_done = plan.interp is not None or not {"interp_pb", "interp_flow_pb"} & set(self.methods)
            if box_done and mosaic_done and interp_done:
                break
        return plan

    def available(self, plan):
        methods = []
        for method in self.methods:
            if method.startswith("temporal_box") and plan.box_source is None:
                continue
            if method.startswith("mosaic") and plan.mosaic is None:
                continue
            if method.startswith("interp") and plan.interp is None:
                continue
            methods.append(method)
        return methods

    @staticmethod
    def noise_sigma(target, mask):
        context = target.copy()
        context[mask] = np.median(target[~mask]) if (~mask).any() else 0
        return pair_noise(context)

    def aligned(self, store, plan, frames, crop, mask, target, channel, use_flow):
        """Kaynak kırpıntıları ve geçerlilikleri; istenirse optik akışla mevcut kareye hizalanmış."""
        sources = [store.crop(t, crop, channel) for t in frames]
        valids = [plan.valids[t] for t in frames]
        if not use_flow:
            return sources, valids
        for t in frames:
            # Akış bölge ve problar için ortak pencerede bir kez hesaplanır, burada kırpıntıya kesilir.
            if t not in plan.flows:
                plan.flows[t] = plan.flow_cache.get(t, crop, mask)
        # En yakın piksel: ham sensör gürültüsü korunur (bilinear örnekleme yamayı pürüzsüzleştirip ele veriyordu).
        sources = [warp(source, plan.flows[t], cv2.INTER_NEAREST) for source, t in zip(sources, frames)]
        valids = [warp(v.astype(np.float32), plan.flows[t], cv2.INTER_NEAREST) > 0.5 for v, t in zip(valids, frames)]
        return sources, valids

    # ---- dolgu

    def fill(self, store, method, plan, system, target, crop, mask, channel=None):
        """Maske içi dolgu değerleri (system.ys, system.xs sırasıyla)."""
        ys, xs = system.ys, system.xs
        if method == "spatial":
            return spatial_fill(system, target)
        if method.startswith("temporal_box"):
            source = store.crop(plan.box_source, crop, channel)
            if method == "temporal_box":
                return source[ys, xs]
            ids = np.zeros(system.n, np.int64)
            return system.blend(target, source[None], ids, plan.valids[plan.box_source][None])
        ry0, ry1, rx0, rx1 = plan.local
        if method.startswith("mosaic"):
            order, id_map = plan.mosaic
            full_ids = np.full(mask.shape, -1, np.int32)
            full_ids[ry0:ry1, rx0:rx1] = id_map
            ids = full_ids[ys, xs].astype(np.int64)
            sources, valids = self.aligned(store, plan, order, crop, mask, target, channel, method == "mosaic_flow_pb")
            return system.blend(target, np.stack(sources), ids, np.stack(valids))
        # Çift yönlü ara değer: her piksel, açıldığı son geçmiş ve ilk gelecek karenin zamana göre ağırlıklı
        # ortalaması. Arka plan zamanla yumuşak değişiyorsa birinci dereceden değişim birbirini götürür.
        (past_order, past_ids), (future_order, future_ids), ring_mask = plan.interp
        use_flow = method == "interp_flow_pb"
        value = np.zeros(mask.shape, np.float32)
        defined = np.zeros(mask.shape, bool)
        sides = []
        for order, id_map in ((past_order, past_ids), (future_order, future_ids)):
            side_value = np.zeros(mask.shape, np.float32)
            side_time = np.zeros(mask.shape, np.float32)
            side_ok = np.zeros(mask.shape, bool)
            if order:
                full_ids = np.full(mask.shape, -1, np.int32)
                full_ids[ry0:ry1, rx0:rx1] = id_map
                sources, valids = self.aligned(store, plan, order, crop, mask, target, channel, use_flow)
                for k, t in enumerate(order):
                    take = full_ids == k
                    side_value[take] = sources[k][take]
                    side_time[take] = abs(t - plan.center)
                    side_ok[take] = valids[k][take]
            sides.append((side_value, side_time, side_ok))
        (past_value, past_time, past_ok), (future_value, future_time, future_ok) = sides
        both = past_ok & future_ok
        total = np.maximum(past_time + future_time, 1e-6)
        value[both] = (future_time[both] * past_value[both] + past_time[both] * future_value[both]) / total[both]
        value[past_ok & ~future_ok] = past_value[past_ok & ~future_ok]
        value[future_ok & ~past_ok] = future_value[future_ok & ~past_ok]
        defined |= past_ok | future_ok
        # İki karenin ağırlıklı ortalaması gürültüyü azaltır, yama çevresinden pürüzsüz görünür; kaybolan
        # gürültü (varyansın 2w(1-w) payı) geri eklenir.
        if both.any():
            weight = future_time[both] / total[both]
            lost = np.sqrt(np.clip(2 * weight * (1 - weight), 0, None))
            rng = np.random.default_rng(int(both.sum()))
            value[both] += rng.normal(0, 1, int(both.sum())).astype(np.float32) * lost * self.noise_sigma(target, mask)
        ids = np.zeros(system.n, np.int64)
        return system.blend(target, value[None], ids, defined[None])

    # ---- değerlendirme

    def probe_crops(self, center, crop, mask, gray, others):
        """Bölgenin yakınında objesiz, aynı biçimli prob kırpıntıları ve her birinin yapı oranı.

        Bölge bulut kenarındayken problar düz gökyüzüne düşerse her aday orada iyi görünür ve puan yanılır.
        Bu yüzden çevresi bölgeninki kadar yapılı problar önce seçilir (yakından uzağa), yetmezse en
        benzerleri. Oran = (bölge yapısı + gürültü) / (prob yapısı + gürültü).
        """
        cy0, cy1, cx0, cx1 = crop
        ys, xs = np.nonzero(mask)
        mh, mw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        occupied = cv2.dilate(self.frame_occupancy(center).astype(np.uint8),
                              np.ones((2 * PROBE_PAD + 1, 2 * PROBE_PAD + 1), np.uint8)) > 0
        width = max(4, int(0.35 * min(mh, mw)))
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        ring = (distance >= 1) & (distance <= width)
        target = gray[cy0:cy1, cx0:cx1]
        usable = np.ones(mask.shape, bool)
        for oy0, oy1, ox0, ox1 in others:
            usable[max(0, oy0 - cy0 - PROBE_PAD):max(0, oy1 - cy0 + PROBE_PAD),
                   max(0, ox0 - cx0 - PROBE_PAD):max(0, ox1 - cx0 + PROBE_PAD)] = False
        noise = pair_noise(target)
        own = structure_level(target, ring & usable)
        offsets = []
        for step in PROBE_STEPS:
            for sy, sx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                offsets.append((int(round(sy * step * mh)), int(round(sx * step * mw))))
        candidates = []
        for dy, dx in offsets:
            py0, py1, px0, px1 = cy0 + dy, cy1 + dy, cx0 + dx, cx1 + dx
            if py0 < 0 or px0 < 0 or py1 > self.height or px1 > self.width:
                continue
            if occupied[py0:py1, px0:px1][mask].any():
                continue
            level = structure_level(gray[py0:py1, px0:px1], ring & ~occupied[py0:py1, px0:px1])
            ratio = 1.0 if own is None or level is None else (own + noise) / (level + noise)
            candidates.append(((py0, py1, px0, px1), ratio, math.hypot(dy, dx)))
        limit = math.log(STRUCTURE_MATCH)
        matched = sorted((c for c in candidates if abs(math.log(c[1])) <= limit), key=lambda c: c[2])
        rest = sorted((c for c in candidates if abs(math.log(c[1])) > limit), key=lambda c: abs(math.log(c[1])))
        chosen = (matched + rest)[:self.args.probes]
        return [c[0] for c in chosen], [c[1] for c in chosen]

    def self_probe_errors(self, store, center, crop, mask, system):
        """Uzamsal dolgu zamana bağlı değildir: bölgenin kendi yerinde, obje oradan çekildikten sonraki ve
        gelmeden önceki ilk karede denenir. Aynı yer olduğu için yapısı bölgeyi en iyi temsil eden sınamadır."""
        cy0, cy1, cx0, cx1 = crop
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ys, xs = np.nonzero(grown)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        first = max(0, center - self.args.search - self.args.margin)
        last = min(len(self.frames) - 1, center + self.args.search)
        valid = trail_valid(self.occupancy(first, last, (cy0 + y0, cy0 + y1, cx0 + x0, cx0 + x1)), self.args.margin)
        free = (valid | ~grown[y0:y1, x0:x1]).all(axis=(1, 2))
        errors = []
        for sign in (-1, 1):
            t = center + sign
            while first <= t <= last and not free[t - first]:
                t += sign
            if first <= t <= last:
                truth = store.crop(t, crop)
                floor = self.noise_floor(store, t, crop, mask, truth)
                values = spatial_fill(system, truth)
                errors.append(float(np.abs(values - truth[system.ys, system.xs]).mean()) / floor)
        return errors

    def noise_floor(self, store, center, crop, mask, target):
        """Hatanın bölüneceği ölçek: bölgenin yakın bir objesiz kareyle farkı (kusursuz zamansal kopyanın hatası)
        ile yerel parlaklık aralığının görünürlük payından büyük olanı. Gürültüsü çok düşük sensörde gözle
        görülmeyen küçük farklar böylece kötü sayılmaz."""
        context = target[~mask] if (~mask).any() else target.ravel()
        low, high = np.percentile(context, (1, 99))
        visible = VISIBLE_FRACTION * float(high - low)
        first = max(0, center - NOISE_FLOOR_FRAMES)
        last = min(len(self.frames) - 1, center + NOISE_FLOOR_FRAMES)
        occupied = self.occupancy(first, last, crop)
        for offset in range(1, NOISE_FLOOR_FRAMES + 1):
            for t in (center - offset, center + offset):
                if first <= t <= last and not occupied[t - first][mask].any():
                    return max(float(np.abs(store.crop(t, crop)[mask] - target[mask]).mean()), visible, 1e-3)
        return max(1.6 * pair_noise(target), visible)

    def quality(self, result, mask, members, others_local, noise):
        """Her obje için kalıntı (kapatılamayan oran) ve bölge için dikiş oranı.

        Kalıntı iki yoldan ölçülür: dolgu ve etiket çevresinde arka plana uymayan piksel oranı, ve maskenin
        hemen dışındaki halkalarda kalan hale (halka medyanı uzak halkaların oynamasını aşıyorsa).
        """
        outside = ~mask
        for oy0, oy1, ox0, ox1 in others_local:
            outside[max(0, oy0 - PROBE_PAD):max(0, oy1 + PROBE_PAD), max(0, ox0 - PROBE_PAD):max(0, ox1 + PROBE_PAD)] = False
        distance = np.rint(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)).astype(np.int32)
        reference = outside & (distance >= 6) & (distance <= 16)
        if reference.sum() < 20:
            reference = outside
        residual = detrend(result, reference)
        _, spread = robust_stats(residual[reference]) if reference.any() else (0.0, noise)
        # Kalıntı objenin işaretini taşır (sıcak obje parlak iz bırakır); bölgedeki objeler aynı işaretliyse
        # yalnız o yön sayılır, böylece bulut içindeki karanlık boşluk kalıntı sanılmaz.
        polarity = getattr(self, "region_polarity", None)
        anomaly = (residual * polarity if polarity else np.abs(residual)) > RESIDUAL_Z * (spread + noise)
        ys, xs = np.nonzero(mask)
        mask_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        signals, ring_spread = ring_profile(residual, distance, outside, mask_box, range(1, 4), range(6, 17), polarity)
        excess = 0.0
        for d in range(1, 4):
            if d in signals:
                threshold = self.halo_threshold(ring_spread, noise, outside & (distance == d))
                excess = max(excess, (signals[d] / threshold - 1) / 4)
        excess = float(np.clip(excess, 0, 1)) * LEFTOVER_BAD
        leftovers = []
        for y0, y1, x0, x1 in members:
            zone = np.zeros(mask.shape, bool)
            zone[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
            zone &= mask | outside
            leftovers.append(max(float(anomaly[zone].mean()) if zone.any() else 0.0, excess))
        inner_steps, outer_steps = [], []
        for dy, dx in DIRECTIONS:
            p = mask
            q = np.roll(np.roll(mask, -dy, axis=0), -dx, axis=1)
            boundary = p & ~q
            boundary[[0, -1], :] = False
            boundary[:, [0, -1]] = False
            by, bx = np.nonzero(boundary)
            qy, qx = by + dy, bx + dx
            ry, rx = qy + dy, qx + dx
            keep = (ry >= 0) & (ry < mask.shape[0]) & (rx >= 0) & (rx < mask.shape[1])
            by, bx, qy, qx, ry, rx = by[keep], bx[keep], qy[keep], qx[keep], ry[keep], rx[keep]
            keep = outside[qy, qx] & outside[ry, rx]
            inner_steps.append(np.abs(result[by[keep], bx[keep]] - result[qy[keep], qx[keep]]))
            outer_steps.append(np.abs(result[qy[keep], qx[keep]] - result[ry[keep], rx[keep]]))
        inner, outer = np.concatenate(inner_steps), np.concatenate(outer_steps)
        seam = float(inner.mean() / max(outer.mean(), noise * 0.5)) if inner.size else 1.0
        return leftovers, seam

    @staticmethod
    def score(error, leftover, seam, texture=1.0):
        q_error = 1.0 if error is None else float(np.clip((ERROR_BAD - error) / (ERROR_BAD - ERROR_GOOD), 0, 1))
        q_left = float(np.clip(1 - leftover / LEFTOVER_BAD, 0, 1))
        q_seam = float(np.clip((SEAM_BAD - seam) / (SEAM_BAD - SEAM_GOOD), 0, 1))
        q_texture = float(np.clip((TEXTURE_BAD - texture) / (TEXTURE_BAD - TEXTURE_GOOD), 0, 1))
        value = min(q_error, q_left, q_seam, q_texture)
        return min(value, NO_PROBE_SCORE_CAP) if error is None else value

    def category(self, value):
        good, middle = self.args.thresholds
        return "iyi" if value >= good else "orta" if value >= middle else "kotu"

    # ---- bölge

    def solve_region(self, store, center, result, gray, rect, mask, members, others, noise):
        crop = self.context_crop(rect)
        cy0, cy1, cx0, cx1 = crop
        y0, y1, x0, x1 = rect
        local = np.zeros((cy1 - cy0, cx1 - cx0), bool)
        local[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = mask
        system = PoissonSystem(local)
        probes, ratios = self.probe_crops(center, crop, local, gray, others)
        flow_cache = FlowCache(self, store, center, gray, [crop] + probes)
        plan = self.make_plan(center, crop, local)
        plan.flow_cache = flow_cache
        methods = self.available(plan)

        errors = {method: [] for method in methods}
        for probe in probes:
            truth = plane(gray[probe[0]:probe[1], probe[2]:probe[3]])
            floor = self.noise_floor(store, center, probe, local, truth)
            probe_plan = self.make_plan(center, probe, local, transplant=crop)
            probe_plan.flow_cache = flow_cache
            for method in methods:
                if method not in self.available(probe_plan):
                    errors[method].append(np.inf)
                    continue
                values = self.fill(store, method, probe_plan, system, truth, probe, local)
                errors[method].append(float(np.abs(values - truth[system.ys, system.xs]).mean()) / floor)
        # Bölge problardan daha yapılıysa prob hatası bölgeye taşınırken büyütülür; uzamsal dolgu ayrıca
        # bölgenin kendi yerinde sınanır ve iki tahminin kötüsü alınır.
        for name in errors:
            errors[name] = [transfer_error(e, r) for e, r in zip(errors[name], ratios)]
        if errors.get("spatial"):
            own_errors = self.self_probe_errors(store, center, crop, local, system)
            if own_errors:
                errors["spatial"] = [max(float(np.median(errors["spatial"])), float(np.median(own_errors)))]
        median_errors = {m: float(np.median(e)) for m, e in errors.items() if e and np.isfinite(np.median(e))}
        if median_errors:
            method = min(median_errors, key=median_errors.get)
            error = median_errors[method]
        else:
            # Prob yoksa zamansal kaynak varsa mozaik, yoksa uzamsal.
            method = next((m for m in ("interp_flow_pb", "mosaic_flow_pb", "mosaic_pb", "temporal_box_pb", "temporal_box")
                           if m in methods),
                          "spatial")
            error = None

        target_gray = gray[cy0:cy1, cx0:cx1].astype(np.float32)
        gray_values = self.fill(store, method, plan, system, target_gray, crop, local)
        # Doku denetimi: dolgu çevresinde olmayan bir yapı içeriyorsa (uzak kareden gelen farklı bulut parçası
        # gibi) piksel hatası küçük olsa da göze batar; sıradaki en iyi adaya geçilir.
        texture = texture_ratio(target_gray, local, gray_values, system.ys, system.xs)
        if texture > TEXTURE_VETO:
            for alternative in sorted(median_errors, key=median_errors.get):
                if alternative == method:
                    continue
                values = self.fill(store, alternative, plan, system, target_gray, crop, local)
                alternative_texture = texture_ratio(target_gray, local, values, system.ys, system.xs)
                if alternative_texture < texture:
                    method, error, gray_values, texture = alternative, median_errors[alternative], values, alternative_texture
                if texture <= TEXTURE_VETO:
                    break
        channels = [None] if result.ndim == 2 else list(range(min(result.shape[2], 3)))
        lo, hi = float(target_gray.min()), float(target_gray.max())
        sources_seen = [store.crop(t, crop) for t in self.plan_sources(plan, method)]
        for source in sources_seen:
            lo, hi = min(lo, float(source.min())), max(hi, float(source.max()))
        values_by_channel = {}
        for channel in channels:
            if channel is None:
                values = gray_values
            else:
                target = plane(result[cy0:cy1, cx0:cx1], channel)
                values = self.fill(store, method, plan, system, target, crop, local, channel)
            if np.issubdtype(self.dtype, np.integer):
                limits = np.iinfo(self.dtype)
                values = np.clip(np.rint(values), max(limits.min, math.floor(lo)), min(limits.max, math.ceil(hi)))
            values_by_channel[channel] = values

        filled_gray = target_gray.copy()
        filled_gray[system.ys, system.xs] = values_by_channel[None] if result.ndim == 2 else gray_values
        members_local = [(my0 - cy0, my1 - cy0, mx0 - cx0, mx1 - cx0) for my0, my1, mx0, mx1 in members]
        others_local = [(oy0 - cy0, oy1 - cy0, ox0 - cx0, ox1 - cx0) for oy0, oy1, ox0, ox1 in others]
        leftovers, seam = self.quality(filled_gray, local, members_local, others_local, noise)
        return {"crop": crop, "system": system, "values": values_by_channel, "method": method, "error": error,
                "errors": median_errors, "probes": len(probes), "leftovers": leftovers, "seam": seam,
                "filled_gray": filled_gray, "texture": texture}

    @staticmethod
    def plan_sources(plan, method):
        if method.startswith("temporal_box"):
            return [plan.box_source]
        if method.startswith("mosaic"):
            return list(plan.mosaic[0])
        if method.startswith("interp"):
            return list(plan.interp[0][0]) + list(plan.interp[1][0])
        return []

    # ---- büyük bölge nakli

    def transplant(self, store, center, result, gray, rect, members, others, noise):
        """Kötü bölge için büyütülmüş pencereyi, kenarında en iyi uyuşan tamamen boş kareden Poisson ile taşır."""
        y0, y1, x0, x1 = rect
        best = None
        for factor in TRANSPLANT_FACTORS:
            half_h = max((y1 - y0) * factor / 2, (y1 - y0) / 2 + 8)
            half_w = max((x1 - x0) * factor / 2, (x1 - x0) / 2 + 8)
            cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
            wy0, wy1 = max(0, int(cy - half_h)), min(self.height, int(math.ceil(cy + half_h)))
            wx0, wx1 = max(0, int(cx - half_w)), min(self.width, int(math.ceil(cx + half_w)))
            if any(oy0 < wy1 and oy1 > wy0 and ox0 < wx1 and ox1 > wx0 for oy0, oy1, ox0, ox1 in others):
                continue
            crop = (max(0, wy0 - TRANSPLANT_RING), min(self.height, wy1 + TRANSPLANT_RING),
                    max(0, wx0 - TRANSPLANT_RING), min(self.width, wx1 + TRANSPLANT_RING))
            cy0, cy1, cx0, cx1 = crop
            mask = np.zeros((cy1 - cy0, cx1 - cx0), bool)
            mask[wy0 - cy0:wy1 - cy0, wx0 - cx0:wx1 - cx0] = True
            if mask.all():
                continue
            target = gray[cy0:cy1, cx0:cx1].astype(np.float32)
            offsets = [o for o in REFERENCE_OFFSETS if o <= self.args.search]
            first = max(0, center - offsets[-1] - self.args.margin)
            last = min(len(self.frames) - 1, center + offsets[-1])
            valid = trail_valid(self.occupancy(first, last, crop), self.args.margin)
            donors = []
            for offset in offsets:
                for t in (center - offset, center + offset):
                    if first <= t <= last and valid[t - first].all():
                        donors.append(t)
                if len(donors) >= TRANSPLANT_DONORS:
                    break
            if not donors:
                continue
            ring = ~mask
            mismatch = []
            for t in donors:
                difference = target[ring] - store.crop(t, crop)[ring]
                mismatch.append(float(np.abs(difference - np.median(difference)).mean()))
            donor = donors[int(np.argmin(mismatch))]
            # Kenar uyuşmazlığı gürültü tabanının birkaç katıysa sahne değişmiştir, nakil tutarsız görünür.
            if min(mismatch) > TRANSPLANT_MATCH * 1.6 * noise:
                continue
            system = PoissonSystem(mask)
            ids = np.zeros(system.n, np.int64)
            everywhere = np.ones((1,) + mask.shape, bool)
            gray_values = system.blend(target, store.crop(donor, crop)[None], ids, everywhere)
            filled = target.copy()
            filled[system.ys, system.xs] = gray_values
            members_local = [(my0 - cy0, my1 - cy0, mx0 - cx0, mx1 - cx0) for my0, my1, mx0, mx1 in members]
            leftovers, seam = self.quality(filled, mask, members_local, [], noise)
            detail = filled - cv2.GaussianBlur(filled, (0, 0), 1.5)
            texture = float(np.std(detail[mask]) / max(np.std(detail[~mask]), noise * 0.3))
            texture = max(texture, 1 / max(texture, 1e-6))
            q_left = float(np.clip(1 - max(leftovers or [0.0]) / LEFTOVER_BAD, 0, 1))
            q_seam = float(np.clip((SEAM_BAD - seam) / (SEAM_BAD - SEAM_GOOD), 0, 1))
            q_texture = float(np.clip((TEXTURE_BAD - texture) / (TEXTURE_BAD - TEXTURE_GOOD), 0, 1))
            value = min(q_left, q_seam, q_texture, TRANSPLANT_CAP)
            if best is None or value > best["score"]:
                best = {"score": value, "crop": crop, "system": system, "donor": donor, "filled": filled,
                        "gray_values": gray_values, "seam": seam, "leftovers": leftovers}
        if best is None:
            return None
        cy0, cy1, cx0, cx1 = best["crop"]
        system = best["system"]
        ids = np.zeros(system.n, np.int64)
        everywhere = np.ones((1,) + best["filled"].shape, bool)
        values = {}
        for channel in ([None] if result.ndim == 2 else list(range(min(result.shape[2], 3)))):
            target = plane(result[cy0:cy1, cx0:cx1], channel)
            donor = store.crop(best["donor"], best["crop"], channel)
            channel_values = best["gray_values"] if channel is None else system.blend(target, donor[None], ids, everywhere)
            if np.issubdtype(self.dtype, np.integer):
                limits = np.iinfo(self.dtype)
                low = min(float(target.min()), float(donor.min()))
                high = max(float(target.max()), float(donor.max()))
                channel_values = np.clip(np.rint(channel_values), max(limits.min, math.floor(low)),
                                         min(limits.max, math.ceil(high)))
            values[channel] = channel_values
        best["values"] = values
        best["ys"], best["xs"] = system.ys + cy0, system.xs + cx0
        return best

    # ---- kare

    def process_frame(self, store, center):
        raw = store.raw(center)
        labels = self.boxes[center]
        if not labels:
            return raw, [], "iyi", []
        gray = plane(raw)
        noise = pair_noise(gray)
        unit = [scale_box(box, 1.0, 0, self.width, self.height) for box in labels]
        objects, polarities = [], {}
        for k, box in enumerate(labels):
            others = [u for j, u in enumerate(unit) if j != k]
            polarities[k] = self.object_polarity(gray, box, noise)
            pad = self.halo_pad(store, center, gray, box, others, noise, polarities[k])
            final, scale = self.final_box(box, pad)
            objects.append((k, box, scale, final))
        union = np.zeros((self.height, self.width), np.uint8)
        for _, _, _, (x0, y0, x1, y1) in objects:
            union[y0:y1, x0:x1] = 1
        count, components, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
        result = raw.copy()
        work_gray = gray.copy()
        rows, regions = [], []
        for c in range(1, count):
            x0, y0, w, h = stats[c][:4]
            rect = (y0, y0 + h, x0, x0 + w)
            mask = components[y0:y0 + h, x0:x0 + w] == c
            inside = [o for o in objects if (components[o[3][1]:o[3][3], o[3][0]:o[3][2]] == c).any()]
            members = [scale_box(o[1], COVERAGE_SCALE, 0, self.width, self.height) for o in inside]
            members = [(my0, my1, mx0, mx1) for mx0, my0, mx1, my1 in members]
            others = [(o[3][1], o[3][3], o[3][0], o[3][2]) for o in objects if o not in inside]
            signs = {polarities[o[0]] for o in inside}
            self.region_polarity = signs.pop() if len(signs) == 1 else None
            solved = self.solve_region(store, center, result, work_gray, rect, mask, members, others, noise)
            cy0, cy1, cx0, cx1 = solved["crop"]
            ys, xs = solved["system"].ys + cy0, solved["system"].xs + cx0
            for channel, values in solved["values"].items():
                if channel is None:
                    result[ys, xs] = values.astype(self.dtype)
                else:
                    result[ys, xs, channel] = values.astype(self.dtype)
            work_gray[cy0:cy1, cx0:cx1] = solved["filled_gray"]
            leftover = max(solved["leftovers"]) if solved["leftovers"] else 0.0
            value = self.score(solved["error"], leftover, solved["seam"], solved["texture"])
            category = self.category(value)
            method_name = solved["method"]
            visible_defect = leftover > TRANSPLANT_LEFTOVER or solved["seam"] > SEAM_BAD
            if self.args.transplant and category == "kotu" and visible_defect:
                moved = self.transplant(store, center, result, work_gray, rect, members, others, noise)
                if moved is not None and moved["score"] > value:
                    for channel, values in moved["values"].items():
                        if channel is None:
                            result[moved["ys"], moved["xs"]] = values.astype(self.dtype)
                        else:
                            result[moved["ys"], moved["xs"], channel] = values.astype(self.dtype)
                    cy0, cy1, cx0, cx1 = moved["crop"]
                    work_gray[cy0:cy1, cx0:cx1] = moved["filled"]
                    value, category, method_name = moved["score"], self.category(moved["score"]), "transplant"
                    solved["leftovers"], solved["seam"] = moved["leftovers"], moved["seam"]
                    rect = (int(moved["ys"].min()), int(moved["ys"].max()) + 1,
                            int(moved["xs"].min()), int(moved["xs"].max()) + 1)
            regions.append((rect, method_name, value, category))
            for (k, box, scale, final), object_left in zip(inside, solved["leftovers"]):
                mask_box = list(final) if method_name != "transplant" else [rect[2], rect[0], rect[3], rect[1]]
                rows.append({"object": k, "box": [round(v, 1) for v in box], "mask": mask_box, "scale": scale, "region": c,
                             "method": method_name, "error": solved["error"], "probes": solved["probes"],
                             "errors": solved["errors"], "leftover": object_left, "seam": solved["seam"],
                             "score": value, "region_category": category})
        frame_category = max((r[3] for r in regions), key=CATEGORIES.index) if regions else "iyi"
        return result, rows, frame_category, regions


def to_8bit(image, low, high):
    scaled = np.clip((image.astype(np.float32) - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR) if scaled.ndim == 2 else scaled[..., :3].copy()


def confidence_map(shape, regions):
    """255 dokunulmamış, 200 iyi dolgu, 128 orta dolgu, 0 kötü bölge; eğitimde 0 olan yer yok sayılabilir."""
    confidence = np.full(shape, 255, np.uint8)
    for (y0, y1, x0, x1), _, _, category in regions:
        confidence[y0:y1, x0:x1] = np.minimum(confidence[y0:y1, x0:x1], CONFIDENCE_VALUES[category])
    return confidence


def write_preview(path, original, result, regions):
    gray = plane(original)
    low, high = np.percentile(gray, (0.5, 99.8))
    left, right = to_8bit(gray, low, high), to_8bit(plane(result), low, high)
    colors = {"iyi": (0, 200, 0), "orta": (0, 200, 255), "kotu": (0, 0, 255)}
    for (y0, y1, x0, x1), method, value, category in regions:
        cv2.rectangle(left, (x0, y0), (x1 - 1, y1 - 1), colors[category], 1)
        cv2.putText(left, f"{SHORT_NAMES[method]} {value:.2f}", (x0, max(10, y0 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[category], 1)
    write_image(path, cv2.hconcat([left, right]))


def process_chunk(sequence, indices, out_dir, preview_every, keep_originals):
    frame_bytes = max(1, read_image(sequence.frames[0]).nbytes)
    store = FrameStore(sequence.frames, size=int(np.clip(CACHE_BYTES // frame_bytes, 16, 400)))
    sequence.store = store
    records = []
    for center in indices:
        path = sequence.frames[center]
        original = store.raw(center)
        result, rows, category, regions = sequence.process_frame(store, center)
        write_image(out_dir / category / f"bg_{path.name}", result) if rows else shutil.copyfile(
            path, out_dir / category / f"bg_{path.name}")
        if keep_originals:
            shutil.copyfile(path, out_dir / ORIGINAL_SUBDIR / path.name)
        if rows:
            write_image(out_dir / CONFIDENCE_SUBDIR / f"{path.stem}.png", confidence_map(original.shape[:2], regions))
        if preview_every and rows and center % preview_every == 0:
            write_preview(out_dir / PREVIEW_SUBDIR / f"{path.stem}.png", original, result, regions)
        for row in rows:
            row["frame"] = path.name
            row["frame_category"] = category
        records.append((path.name, category, rows))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", help="2_Etiketlenecek_Veri ve 3_Etiketlenmis_Veri_Ciktisi klasörlerinin kökü")
    parser.add_argument("--frames", help="TIFF klasörü, varsayılan <project>/2_Etiketlenecek_Veri/tif")
    parser.add_argument("--images-json", help="varsayılan <project>/3_Etiketlenmis_Veri_Ciktisi/coco-images.json")
    parser.add_argument("--annotations-json", help="varsayılan <project>/3_Etiketlenmis_Veri_Ciktisi/coco-annotations.json")
    parser.add_argument("--labels", help="COCO yerine YOLO .txt klasörü veya frame,x,y,w,h CSV dosyası")
    parser.add_argument("--out", required=True, help="çıktı kökü: iyi/, orta/, kotu/, silinmemis/, onizleme/, rapor")
    parser.add_argument("--scales", default="1.0,1.1,1.2,1.3,1.4", help="denenecek etiket büyütme katsayıları")
    parser.add_argument("--methods", default=",".join(METHODS), help="kullanılacak adaylar, virgülle")
    parser.add_argument("--probes", type=int, default=4, help="bölge başına prob sayısı")
    parser.add_argument("--search", type=int, default=1000, help="zamansal kaynağın en çok uzaklığı (kare)")
    parser.add_argument("--margin", type=int, default=3, help="obje çıktıktan sonra beklenecek kare (ısıl iz)")
    parser.add_argument("--min-offset", type=int, default=1, help="kaynağın hedefe en az uzaklığı (kare)")
    parser.add_argument("--max-sources", type=int, default=6, help="mozaikte en çok kaynak kare")
    parser.add_argument("--occupancy-scale", type=float, default=1.3, help="kaynak seçerken obje kutusu büyütmesi")
    parser.add_argument("--occupancy-pad", type=int, default=6, help="kaynak seçerken ek piksel payı")
    parser.add_argument("--thresholds", default="0.8,0.2", help="iyi ve orta puan eşikleri")
    parser.add_argument("--preview-every", type=int, default=0, help="her N karede bir 8 bit önizleme, 0 kapalı")
    parser.add_argument("--no-originals", action="store_true", help="silinmemis/ kopyalarını yazma")
    parser.add_argument("--footprint", action="store_true",
                        help="deneysel: kutu yerine obje biçimli doluluk (dokulu zeminde zayıf parçaları kaçırabilir)")
    parser.add_argument("--no-transplant", dest="transplant", action="store_false",
                        help="kötü bölgeler için büyük bölge naklini kapat")
    parser.add_argument("--every", type=int, default=1, help="yalnız her N. kareyi işle (hızlı deneme)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.scales = [float(s) for s in args.scales.split(",")]
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(args.methods) - set(METHODS)
    if unknown:
        parser.error(f"bilinmeyen yöntem: {', '.join(sorted(unknown))}")
    args.thresholds = tuple(float(v) for v in args.thresholds.split(","))

    if args.project:
        root = Path(args.project)
        args.frames = args.frames or root / FRAMES_SUBDIR
        args.images_json = args.images_json or root / COCO_SUBDIR / "coco-images.json"
        args.annotations_json = args.annotations_json or root / COCO_SUBDIR / "coco-annotations.json"
    if not args.frames or not (args.labels or (args.images_json and args.annotations_json)):
        parser.error("--project ya da --frames ile birlikte iki COCO dosyası veya --labels verilmeli")

    started = time.time()
    frames = list_frames(args.frames)
    height, width = read_image(frames[0]).shape[:2]
    if args.labels:
        loader = load_csv_labels if Path(args.labels).is_file() else load_yolo_labels
        boxes = loader(args.labels, frames, width, height)
    else:
        boxes = load_coco_labels(args.images_json, args.annotations_json, frames, width, height)
    skipped = sum(frame_boxes is None for frame_boxes in boxes)
    if skipped == len(frames):
        raise ValueError("Hiçbir TIFF adı COCO file_name alanıyla eşleşmedi")
    if skipped:
        print(f"UYARI: COCO'da kaydı olmayan {skipped} TIFF işlenmedi ve kaynak olarak kullanılmadı.")

    out_dir = Path(args.out)
    for sub in CATEGORIES + (PREVIEW_SUBDIR, CONFIDENCE_SUBDIR) + (() if args.no_originals else (ORIGINAL_SUBDIR,)):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    records = []
    # İş parçacığı yerine süreç: numpy dışı Python işleri de paralel koşsun.
    with ProcessPoolExecutor(args.workers) as pool:
        tasks = []
        for sequence_frames, sequence_boxes in split_sequences(frames, boxes):
            sequence = Sequence(sequence_frames, sequence_boxes, args)
            # --every ile yalnız her N. kare işlenir (hızlı deneme); kaynak olarak yine tüm kareler kullanılır.
            indices = list(range(0, len(sequence_frames), args.every))
            step = max(1, math.ceil(len(indices) / (args.workers * 4)))
            for start in range(0, len(indices), step):
                tasks.append(pool.submit(process_chunk, sequence, indices[start:start + step],
                                         out_dir, args.preview_every, not args.no_originals))
        for done, task in enumerate(as_completed(tasks), 1):
            records.extend(task.result())
            print(f"{done}/{len(tasks)} parça", flush=True)

    records.sort()
    with open(out_dir / "rapor.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "frame_category", "object", "x1", "y1", "x2", "y2", "mask_x0", "mask_y0", "mask_x1",
                         "mask_y1", "scale", "region", "method",
                         "score", "region_category", "probe_error", "probes", "leftover", "seam"]
                        + [f"error_{m}" for m in METHODS])
        for name, category, rows in records:
            for row in rows:
                writer.writerow([name, category, row["object"], *row["box"], *row["mask"], row["scale"], row["region"],
                                 row["method"],
                                 round(row["score"], 3), row["region_category"],
                                 "" if row["error"] is None else round(row["error"], 3), row["probes"],
                                 round(row["leftover"], 4), round(row["seam"], 3)]
                                + [round(row["errors"][m], 3) if m in row["errors"] else "" for m in METHODS])
    frame_counts = Counter(category for _, category, _ in records)
    method_counts = Counter(row["method"] for _, _, rows in records for row in rows)
    summary = {"kareler": dict(frame_counts), "yontemler": dict(method_counts),
               "nesne_sayisi": sum(len(rows) for _, _, rows in records), "sure_sn": round(time.time() - started, 1)}
    (out_dir / "ozet.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Kare: {len(records)}  " + "  ".join(f"{c}: {frame_counts.get(c, 0)}" for c in CATEGORIES)
          + f"  yöntemler: {dict(method_counts)}  süre: {summary['sure_sn']} sn")


if __name__ == "__main__":
    main()
