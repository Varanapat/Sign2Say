import json, pathlib

js = json.load(open("tsl_one_s_dataset_i3d.json", encoding="utf-8"))
id2gloss = {i["video_id"]: g["gloss"] for g in js for i in g["instances"]}
have = {p.stem for p in pathlib.Path("data/pose_npy").glob("*.npy")}

print("อยู่ใน JSON แต่ไม่มี .npy:")
for v in sorted(set(id2gloss) - have):
    print(f"  {v}  ({id2gloss[v]})")
print("มี .npy แต่ไม่อยู่ใน JSON:")
for v in sorted(have - set(id2gloss)):
    print(f"  {v}")
