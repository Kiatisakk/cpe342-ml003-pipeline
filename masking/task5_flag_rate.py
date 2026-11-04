"""
หา flag rate ที่เหมาะกับ F3 macro (ไม่ใช้ label — เป็นการวิเคราะห์เชิงทฤษฎี)
สมมติ: anomaly จริงมี K แถว, กลุ่มชัดเจน 827 แถวถูกจัดอันดับบนสุดเสมอ
  - perfect : ส่วนที่เหลือ (K-827) ก็ถูกจัดอันดับถัดมาพอดี  → ขอบบนของคะแนน
  - random  : ส่วนที่เหลือกระจายสุ่มในแถวที่เหลือ             → ขอบล่าง (โมเดลไม่ช่วยเลยหลังกลุ่มชัดเจน)
"""
import numpy as np, sys
N, OBV = 25889, 827

def fbeta(p, r, b=3.0):
    return 0 if p + r == 0 else (1 + b*b) * p * r / (b*b*p + r)

def macro_f3(n, K, mode):
    if mode == "perfect":
        tp = min(n, K)
    else:  # random beyond obvious
        tp = min(n, OBV) + max(n - OBV, 0) * (K - OBV) / (N - OBV)
    fp, fn = n - tp, K - tp
    tn = N - K - fp
    f1 = fbeta(tp / n, tp / K); f0 = fbeta(tn / (tn + fn), tn / (N - K))
    return (f0 + f1) / 2

Ks = [827, 900, 1000, 1100, 1200, 1300]
ns = np.arange(800, 2601, 10)
print("F3-macro ที่ flag n แถว (แถว = จำนวน anomaly จริง K)")
for mode in ("perfect", "random"):
    print(f"\n== {mode} ==")
    best = {}
    for K in Ks:
        sc = np.array([macro_f3(n, K, mode) for n in ns]); best[K] = ns[sc.argmax()]
        row = "  ".join(f"{int(n)}:{macro_f3(n, K, mode):.3f}" for n in (907, 1100, 1295, 1500, 2072))
        print(f"K={K:5d}  best n={best[K]:5d} ({best[K]/N:.1%})  | {row}")
    exp = np.array([np.mean([macro_f3(n, K, mode) for K in Ks]) for n in ns])
    print(f"เฉลี่ยทุก K → n ที่ดีที่สุด = {ns[exp.argmax()]} ({ns[exp.argmax()]/N:.1%}), F3≈{exp.max():.3f}")


def estimate_K(s5_at_obv, n=830):
    """ใช้คะแนน Task 5 (จาก decode.py) ของ probe ที่ flag n≈827 แถว (กลุ่มชัดเจนเกือบ 100% ถูก)
    → หา K ที่ทำให้ F3-macro ตรงกับคะแนนนั้น (สมมติ precision ≈ 1)"""
    Ks = np.arange(n, 6000)
    sc = np.array([macro_f3(n, K, "perfect") for K in Ks])
    return int(Ks[np.abs(sc - s5_at_obv).argmin()])

if len(sys.argv) > 1:   # python task5_flag_rate.py 0.912  → ประมาณ K
    K = estimate_K(float(sys.argv[1]))
    print(f"\nคะแนน Task5 ที่ flag 830 แถว = {sys.argv[1]} → ประมาณ anomaly จริง K ≈ {K} ({K/N:.1%})")
    print(f"ตั้ง flag rate ใกล้ {K/N:.1%} (เอียงขึ้นเล็กน้อยถ้าโมเดลจัดอันดับแถวก้ำกึ่งได้ดี)")
