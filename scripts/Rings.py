import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class tower_set:
    def __init__(self, lowest_ring=0, n_rings=5, remaining_rings=None):
        if remaining_rings is None:
            self.remaining_rings = [r + 1 for r in range(n_rings)]
        else:
            self.remaining_rings=remaining_rings
        self.n_rings=n_rings
        self.lowest_ring=lowest_ring
        if self.lowest_ring==self.n_rings:
            self.total=1
        else:
            sub_towers = [tower_set(lowest_ring=max(r,self.lowest_ring+1), n_rings=self.n_rings, remaining_rings=[p for p in self.remaining_rings if p != r]).total
                          for r in self.remaining_rings]
            self.total = sum(sub_towers)

    def print_total(self):
        print(self.total)

max_n=12
coms=[tower_set(n_rings=i+1).total for i in range(max_n)]
rings=[i+1 for i in range(max_n)]
plt.plot(rings,coms)
plt.xlabel('Number of rings')
plt.ylabel('Combinations')
plt.yscale('log')
plt.savefig('rings')
plt.show()