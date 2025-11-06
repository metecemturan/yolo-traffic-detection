import os
import json
import shutil
from collections import defaultdict

COCO_ROOT = r"C:\Users\metec\OneDrive\Masaüstü\coco2017" 
TRAIN_JSON = r"C:\Users\metec\OneDrive\Masaüstü\coco2017\annotations\filtered_instances_train2017_3class.json"
VAL_JSON   = r"C:\Users\metec\OneDrive\Masaüstü\coco2017\annotations\filtered_instances_val2017_3class.json"

OUT_ROOT = r"C:\Users\metec\OneDrive\Masaüstü\YAP470_Dataset"
#YAP470-> train & val-> images & labels -> .txt (class_id x_center y_center width height)


COCO_TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017")
COCO_VAL_IMG_DIR   = os.path.join(COCO_ROOT, "val2017")

TARGET_CLASS_ORDER = ["car", "bus", "motorcycle"]

# {'car':0, 'bus':1, 'motorcycle':2}
YOLO_CLASS_TO_ID = {name: i for i, name in enumerate(TARGET_CLASS_ORDER)}  

def make_dirs(split_root):
    os.makedirs(os.path.join(split_root, "images"))
    os.makedirs(os.path.join(split_root, "labels"))
    

def to_yolo_bbox(bbox, img_w, img_h):
    # COCO: [x_min, y_min, w, h] → YOLO: [x_c/img_w, y_c/img_h, w/img_w, h/img_h]
    x, y, w, h = bbox

    # negatif, sıfır genişlik/yükseklik 
    if w <= 0 or h <= 0:
        return None

    # x_center ve y_center hesapla
    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    w /= img_w
    h /= img_h

    #normalizasyon
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    # eğer kutu görüntü dışında kalmışsa None döndür
    if w == 0 or h == 0:
        return None

    return x_center, y_center, w, h

def process_split(json_path, coco_img_dir, out_split_dir):
    make_dirs(out_split_dir)
    
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    
    #{3: "car", 4: "motorcycle", 6: "bus"}
    catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    
    
    #cocoid:yoloid -> {3: 0, 6: 1, 4: 2}
    catid_to_yolo = {}
    for cid, cname in catid_to_name.items():
        if cname not in YOLO_CLASS_TO_ID:
            continue
        catid_to_yolo[cid] = YOLO_CLASS_TO_ID[cname]



    '''anns_by_img = {
        1: [ {car box}, {motor box} ],
        2: [ {bus box} ]
        }'''
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann["category_id"] not in catid_to_yolo:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    for img in coco["images"]:
        img_id = img["id"]
        file_name = img["file_name"]
        width, height = img["width"], img["height"]

        ann_list = anns_by_img.get(img_id, [])
        if not ann_list:
            continue  

        # label dosyası yolu -> labels/000001.txt
        stem = os.path.splitext(file_name)[0]
        label_path = os.path.join(out_split_dir, "labels", f"{stem}.txt")

        #her boxun txtye yazimi -> 2 0.625000 0.300000 0.312500 0.208333 -> motorcycle
        with open(label_path, "w", encoding="utf-8") as lf:
            for ann in ann_list:
                yolo_id = catid_to_yolo[ann["category_id"]]
                box = to_yolo_bbox(ann["bbox"], width, height)
                if box is None:
                    continue
                x_c, y_c, w, h = box
                lf.write(f"{yolo_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        # Görseli kopyala
        src_img = os.path.join(coco_img_dir, file_name)
        dst_img = os.path.join(out_split_dir, "images", file_name)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        try:
            shutil.copy2(src_img, dst_img)
        except FileNotFoundError:
            print(f"[WARN] Görsel bulunamadı: {file_name}")
            continue

    print(f"{out_split_dir} tamamlandı.")

def main():
    # TRAIN 
    process_split(
        json_path=TRAIN_JSON,
        coco_img_dir=COCO_TRAIN_IMG_DIR,
        out_split_dir=os.path.join(OUT_ROOT, "train")
    )

    # VAL
    process_split(
        json_path=VAL_JSON,
        coco_img_dir=COCO_VAL_IMG_DIR,
        out_split_dir=os.path.join(OUT_ROOT, "val")
    )

    print("Dataset oluşturma tamamlandı")

if __name__ == "__main__":
    main()
