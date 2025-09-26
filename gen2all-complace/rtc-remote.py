import sys as Z, base64 as B, itertools as I, random as R

Q = lambda s,k: bytes([a ^ k for a in s])
J = lambda n: R.randint(1,255)
K = lambda d: B.b64encode(d).decode()

def G(p):
    with open(p,'rb') as F: return F.read()

def W(t):
    R.seed(len(t)*7 + sum(t)%313)
    k = J( )
    u = Q(t,k)
    v = list(u)
    R.shuffle(v)
    return K(bytes(v))

def V(a):
    # chunk-reverse & interleave
    m = [a[i:i+R.randint(2,7)] for i in range(0,len(a),R.randint(3,9))]
    R.shuffle(m)
    return ''.join(''.join(reversed(x)) for x in m)

def H():
    if len(Z.argv)<2:
        Z.stderr.write("usage: python rtc-remote.py <file>\n")
        Z.exit(1)
    p = Z.argv[1]
    d = G(p)
    s = W(d)
    y = V(s)
    Z.stdout.write(y + "\n")

if __name__=='__main__':
    H()
