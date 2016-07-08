class C(object):
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x

    def setx(self, value):
        self._x = value

    def delx(self):
        del self._x

    x = property(getx, setx, delx, "I'm the 'x' property.")


import numpy as np

s=5
k=3
X=np.identity(s)
Y=np.zeros((s,k))
Z=np.concatenate((X,Y), axis=1)
W=np.insert(Z, 3, np.zeros(s),axis=1)
#print np.dot(W,W.T)
#print W

W = np.array([[1,5, 10],[3, 10, 18], [8, 5, 11]], dtype=float)
#print W
#eta = np.zeros(2)
#eta[1]=1
#R=np.dot(W,eta)
#print np.dot(R, eta.T)

Sigma=np.dot(W,W.T)
#print Sigma
#print np.linalg.inv(Sigma)
mu = np.random.normal(size=Sigma.shape[0])
j=0
mu[j]=0
#rint np.dot(Sigma[:,j].T, np.linalg.inv(Sigma))
#print np.dot(np.dot(Sigma[:,j].T, np.linalg.inv(Sigma)), mu)


a=np.array([1,2,3,4,-3])
#print np.clip(a, 0, np.inf)

d={'Jelena':1, 'Bojana':2}
def proba(Jelena=0, Bojana=0):
    return Jelena+1

#print proba(**d)
#print map(proba,**d)


n=5
eta=1
indices = np.random.choice(n, size=(n,), replace = True)
print indices
indices1 = [i if np.random.uniform(0,1,1)<eta else indices[i] for i in range(n)]

print indices1
for i in range(n):
    if (np.random.uniform(0,1,1)<eta):
        indices[i]=i
        print i
print indices











