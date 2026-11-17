import base64, io, json, os, sys, time, urllib.request, csv, argparse
AP=argparse.ArgumentParser(); AP.add_argument("--port",type=int,default=8300)
AP.add_argument("--sec",type=float,default=8.0); AP.add_argument("--verify",action="store_true")
A=AP.parse_args()
row=[r for r in csv.DictReader(io.open("experiments/results/f5_prompts.csv",encoding="utf-8"))
     if r["speaker"]=="61"][0]
src=row["ref_path"]                      # a .flac, so this tests decoding too
raw=open(src,"rb").read()
print("uploading %s (%.0f KB, %s)"%(os.path.basename(src),len(raw)/1024,src.rsplit('.',1)[-1]))
def post(path,obj,timeout=900):
    rq=urllib.request.Request(f"http://127.0.0.1:{A.port}{path}",
        data=json.dumps(obj).encode(),headers={"Content-Type":"application/json"})
    try: return json.loads(urllib.request.urlopen(rq,timeout=timeout).read())
    except urllib.error.HTTPError as e: return {"error":"HTTP %d %s"%(e.code,e.read()[:300].decode())}
t=time.time()
r=post("/api/protect",{"audio":"data:audio/flac;base64,"+base64.b64encode(raw).decode(),"sec":A.sec})
print("protect: %.0f s"%(time.time()-t))
if "error" in r: print("  ERROR:",r["error"]); sys.exit(1)
print("  quality:",r["quality"],"craft_s",r["craft_s"])
print("  audio:",list(r["audio"].values()))
if A.verify:
    t=time.time(); v=post("/api/verify",{"run_id":r["run_id"]})
    print("verify: %.0f s"%(time.time()-t))
    if "error" in v: print("  ERROR:",v["error"]); sys.exit(1)
    for x in v["rows"]:
        print("  %-22s tau %.3f | unprot %.3f %s | prot %.3f %s"%(x["encoder"],x["threshold"],
            x["sim_unprotected"],"CLONED" if x["cloned_unprotected"] else "failed",
            x["sim_protected"],"CLONED" if x["cloned_protected"] else "failed"))
