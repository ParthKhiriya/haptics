"""
extract_and_inspect.py
----------------------
Reads the TUM haptic .mat files (MATLAB v5, struct `finalMaterialRecording`),
keeps ONLY the small 1-D signals (drops the huge images), prints a report,
runs quick audio<->acceleration diagnostics, and saves a compact file.

Usage:
    python extract_and_inspect.py /path/to/one_file.mat
    python extract_and_inspect.py /path/to/dataset_root --out signals_compact.npz --csv diagnostics.csv

Needs: numpy, scipy.   (If a file is MATLAB v7.3 / HDF5: pip install h5py, see note below.)
Run one file first and send me the printed output.
"""
import argparse, os, sys, csv
import numpy as np
from scipy import io as sio
from scipy import signal

SIG_KEYS = ["accelDFT", "sound", "accelDFTSteelTooltip", "soundSteelTooltip",
            "normalForce", "frictionForce"]
META_KEYS = ["materialID", "materialNameEnglish", "labelFinePX", "labelFine",
             "labelCoarse", "labelMedium", "instance", "density"]
FS = 3000.0  # Hz, per the database readme (sliding signals)


def find_structs(mat):
    """Return a list of mat_struct objects found in the loaded .mat dict."""
    out = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, sio.matlab.mat_struct):
            out.append(v)
        elif isinstance(v, np.ndarray) and v.dtype == object:
            out += [x for x in v.ravel() if isinstance(x, sio.matlab.mat_struct)]
    return out


def get(s, name):
    return getattr(s, name, None)


def stats(x):
    x = np.asarray(x, dtype=float).ravel()
    return (f"n={x.size} mean={x.mean():.4g} std={x.std():.4g} "
            f"min={x.min():.4g} max={x.max():.4g} nan={np.isnan(x).sum()}")


def diagnostics(snd, acc):
    """Quick audio-vs-accel relationship measures (same rate, same length)."""
    snd = np.asarray(snd, float).ravel(); acc = np.asarray(acc, float).ravel()
    n = min(snd.size, acc.size); snd, acc = snd[:n], acc[:n]
    snd = snd - snd.mean(); acc = acc - acc.mean()
    r0 = np.corrcoef(snd, acc)[0, 1] if snd.std() > 0 and acc.std() > 0 else np.nan
    # best lag within +-50 ms
    maxlag = int(0.05 * FS)
    xc = signal.correlate(acc, snd, mode="full")
    lags = signal.correlation_lags(acc.size, snd.size, mode="full")
    sel = np.abs(lags) <= maxlag
    best = lags[sel][np.argmax(np.abs(xc[sel]))]
    # magnitude-squared coherence, averaged in bands
    f, c = signal.coherence(snd, acc, fs=FS, nperseg=512)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(np.mean(c[m])) if m.any() else np.nan
    # envelope correlation (rectify + 20 Hz low-pass)
    b, a = signal.butter(2, 20 / (FS / 2))
    es = signal.filtfilt(b, a, np.abs(signal.hilbert(snd)))
    ea = signal.filtfilt(b, a, np.abs(signal.hilbert(acc)))
    renv = np.corrcoef(es, ea)[0, 1]
    return dict(corr0=r0, best_lag_samples=int(best), best_lag_ms=1000 * best / FS,
                coh_10_100=band(10, 100), coh_100_400=band(100, 400),
                coh_400_1000=band(400, 1000), env_corr=float(renv),
                rms_ratio_acc_over_snd=float(acc.std() / (snd.std() + 1e-12)))


def process_file(path, store, rows, verbose):
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        print(f"[!] {path} is MATLAB v7.3 (HDF5). Use h5py: pip install h5py, "
              "then tell me and I will add an h5py branch.")
        return
    structs = find_structs(mat)
    if not structs:
        print(f"[!] No struct found in {path}. Top-level keys: "
              f"{[k for k in mat if not k.startswith('__')]}")
        return
    for i, s in enumerate(structs):
        meta = {k: get(s, k) for k in META_KEYS}
        tag = f"{os.path.basename(path)}#{i}"
        if verbose:
            print(f"\n=== {tag} ===")
            print("fields:", getattr(s, "_fieldnames", "?"))
            print("meta:", {k: (v if np.isscalar(v) or isinstance(v, str) else type(v).__name__)
                            for k, v in meta.items()})
        rec = {"file": tag}
        for k in SIG_KEYS:
            v = get(s, k)
            if v is None:
                continue
            v = np.asarray(v, dtype=float).ravel()
            if verbose:
                print(f"  {k:24s} {stats(v)}")
            store.setdefault(k, []).append(v)
            rec[k] = v
        store.setdefault("_meta", []).append(
            {k: str(meta[k]) for k in META_KEYS} | {"file": tag})
        for a, b_ in [("accelDFT", "sound"), ("accelDFTSteelTooltip", "soundSteelTooltip")]:
            if a in rec and b_ in rec:
                d = diagnostics(rec[b_], rec[a])
                d.update(file=tag, pair=f"{b_}->{a}", materialID=meta.get("materialID"))
                rows.append(d)
                if verbose:
                    print(f"  [{b_} -> {a}] " + ", ".join(
                        f"{k}={v:.3g}" for k, v in d.items() if isinstance(v, (float, int))))
        del s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="signals_compact.npz")
    ap.add_argument("--csv", default="diagnostics.csv")
    ap.add_argument("--max", type=int, default=0, help="stop after N files (0 = all)")
    args = ap.parse_args()

    files = []
    if os.path.isfile(args.path):
        files = [args.path]
    else:
        for r, _, fs in os.walk(args.path):
            files += [os.path.join(r, f) for f in fs if f.lower().endswith(".mat")]
        files.sort()
    if args.max:
        files = files[:args.max]
    print(f"{len(files)} .mat file(s)")

    store, rows = {}, []
    for j, p in enumerate(files):
        process_file(p, store, rows, verbose=(len(files) <= 3 or j < 2))
        if j % 20 == 0 and len(files) > 3:
            print(f"  processed {j+1}/{len(files)}")

    # save compact arrays (only keys whose lengths agree)
    out = {}
    for k, lst in store.items():
        if k == "_meta":
            continue
        lens = {len(x) for x in lst}
        if len(lens) == 1:
            out[k] = np.stack(lst).astype(np.float32)
        else:
            print(f"[!] {k}: varying lengths {sorted(lens)[:5]} -> saved as padded + lengths")
            L = max(lens); arr = np.zeros((len(lst), L), np.float32)
            for i, x in enumerate(lst):
                arr[i, :len(x)] = x
            out[k] = arr; out[k + "_len"] = np.array([len(x) for x in lst])
    if store.get("_meta"):
        out["meta_json"] = np.array([str(m) for m in store["_meta"]])
    if out:
        np.savez_compressed(args.out, **out)
        print(f"saved {args.out}: " + ", ".join(f"{k}{v.shape}" for k, v in out.items() if k != 'meta_json'))
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"saved {args.csv}")
        for pair in sorted({r['pair'] for r in rows}):
            sub = [r for r in rows if r['pair'] == pair]
            m = lambda k: np.nanmean([r[k] for r in sub])
            print(f"\nSUMMARY {pair} (n={len(sub)}): corr0={m('corr0'):.3f} "
                  f"env_corr={m('env_corr'):.3f} coh[10-100]={m('coh_10_100'):.3f} "
                  f"coh[100-400]={m('coh_100_400'):.3f} coh[400-1000]={m('coh_400_1000'):.3f} "
                  f"mean|lag|={np.mean([abs(r['best_lag_ms']) for r in sub]):.2f} ms")


if __name__ == "__main__":
    main()