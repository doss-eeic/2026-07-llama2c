import hashlib, os, subprocess, sys
i = sys.argv.index("--")
env = dict(os.environ, **dict(a.split("=", 1) for a in sys.argv[2:i]))
p = subprocess.run(sys.argv[i + 1:], env=env, capture_output=True, text=True, timeout=300)
open(sys.argv[1], "w").write(p.stderr)
print(hashlib.sha256(p.stdout.encode()).hexdigest()[:16])
