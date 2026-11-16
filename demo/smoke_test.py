import base64, json, urllib.request, time, sys, io, os, csv
sys.path.insert(0, os.getcwd())
rows=list(csv.DictReader(io.open("experiments/results/f5_prompts.csv",encoding="utf-8")))
r=[x for x in rows if x["speaker"]=="61"][0]
import soundfile as sf, numpy as np
def b64(path):
    y,sr=sf.read(path,dtype="float32")
    if y.ndim>1: y=y.mean(1)
    y=y[:int(4*sr)]
    buf=io.BytesIO(); sf.write(buf,y,sr,format="WAV",subtype="PCM_16")
    return "data:audio/wav;base64,"+base64.b64encode(buf.getvalue()).decode()
pub=b64(r["ref_path"]); enr=b64(r["enrol_path"])
body=json.dumps({"publish":pub,"enrol":enr,"steps":60}).encode()
t0=time.time()
req=urllib.request.Request("http://127.0.0.1:8201/api/run",data=body,
                           headers={"Content-Type":"application/json"})
try:
    out=json.loads(urllib.request.urlopen(req,timeout=900).read())
except urllib.error.HTTPError as e:
    print("HTTP",e.code,e.read()[:900].decode()); sys.exit(1)
print("elapsed %.0f s"%(time.time()-t0))
print(json.dumps({k:out[k] for k in ("quality","craft_s","steps") if k in out},indent=1))
for x in out.get("rows",[]):
    print("  %-22s tau %.3f | unprot %.3f %s | prot %.3f %s"%(
        x["encoder"],x["threshold"],x["sim_unprotected"],
        "CLONED" if x["cloned_unprotected"] else "failed",
        x["sim_protected"],"CLONED" if x["cloned_protected"] else "failed"))
print("audio:",list(out.get("audio",{}).values()))
