# HTA RCCL Support Patch (v1)

This directory contains an RCCL support patch set for HolisticTraceAnalysis (HTA).

## Quick Use

```bash
cd /path/to/HolisticTraceAnalysis
git checkout -b rccl-patch-base 99bcc82
git am /path/to/perf-variability-ipdps26/HolisticTraceAnalysis-rccl-patch-v1/rccl-support-v1.patchset.mbox
git log --oneline -n 2
```

After success, you should see these two commits (display order may vary):

- `detect rccl_main_kernel as communication`
- `add code to append rccl collective name to end of traceEvent name`

