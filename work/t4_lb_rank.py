# Run work/t4_lb_consistency.py with a custom candidate list (argv) and extra past LB points.
# sub_v72a_em scored 0.86997 with the 2,115-row T5 (+11 rows, ~+0.00003) -> added as 0.86994 with the base T5.
import sys
s = open("work/t4_lb_consistency.py", encoding="utf-8").read()
s = s.replace('"sub_v59r3_noMLP": .86925}', '"sub_v59r3_noMLP": .86925, "sub_v72a_em": .86994, "sub_v73a_em": .87013}')
i = s.index("CAND = "); j = s.index("\n", i)
exec(s[:i] + "CAND = " + repr(sys.argv[1:]) + s[j:])
