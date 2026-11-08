import json, pathlib, collections

js = json.load(open("tsl_one_s_dataset_i3d.json", encoding="utf-8"))
have = {p.stem for p in pathlib.Path("data/pose_npy").glob("*.npy")}

total, missing = collections.Counter(), collections.Counter()
for g in js:
    for inst in g["instances"]:
        total[inst["split"]] += 1
        if inst["video_id"] not in have:
            missing[inst["split"]] += 1

print("จำนวน .npy ที่มี:", len(have), "/ 4152")
print("ชื่อ split ใน JSON และจำนวน:", dict(total))
print("คลิปที่ยังไม่มี .npy:", dict(missing))
by_signer = collections.Counter(v.split("_")[0] for v in have)
print("จำนวน .npy ต่อ signer:", dict(sorted(by_signer.items())))
for s in ["03", "08", "22", "11", "25", "29"]:  # signer ของ test/val ใน split B
    print(f"  signer {s}: {by_signer.get(s, 0)} ไฟล์")