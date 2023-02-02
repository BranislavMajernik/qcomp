import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from minorminer.busclique import find_clique_embedding
import dwave_networkx as dnx
from pyqubo import Binary, Constraint, Placeholder, Array, LogEncInteger

# weights, values and the maximum weight of the knapsack problem
weights = [1, 3, 7, 9]
values = [10, 2, 3, 6]
max_weight = 10
n=len(values)
items = Array.create('item', shape=n, vartype="BINARY")
knapsack_weight = sum(weights[i] * items[i] for i in range(n))
knapsack_value = sum(values[i] * items[i] for i in range(n))

# create Hamiltonian and model
weight_one_hot = LogEncInteger("weight_one_hot",value_range=(1, max_weight))
Ha = Constraint((weight_one_hot - knapsack_weight)**2,"weight_constraint")
Hb = knapsack_value
lmd = Placeholder("lmd")
H = lmd*Ha - Hb
model = H.compile()

dw_sampler = DWaveSampler(endpoint="https://na-west-1.cloud.dwavesys.com/sapi/v2",token="DEV-a03a6447bd1ad04cd8d5b9102cc9888812bde882",solver="Advantage_system4.1")
graph_size=16
sampler_size=len(model.variables)
p16_working_graph = dnx.pegasus_graph(graph_size,node_list=dw_sampler.nodelist,edge_list=dw_sampler.edgelist)
embedding = find_clique_embedding(sampler_size,p16_working_graph)
sampler = FixedEmbeddingComposite(dw_sampler, embedding)
sampler_kwargs = {"num_reads": 100,"annealing_time": 20,"num_spin_reversal_transforms": 4,"auto_scale": True,"chain_strength": 2.0,"chain_break_fraction": True}

def objective(feed_dict):
    bqm = model.to_bqm(index_label=True, feed_dict=feed_dict)
    bqm.normalize()
    sampleset = sampler.sample(bqm, **sampler_kwargs)
    dec_samples = model.decode_sampleset(sampleset,feed_dict=feed_dict)
    return min(dec_samples, key=lambda x: x.energy)

# search best parameters lmd within [1,2,...,5]
feasible_sols = []
for lmd_value in range(1, 5):
    feed_dict = {'lmd': lmd_value}
    s = objective(feed_dict)
    if not s.constraints(only_broken=True):
        feasible_sols.append(s)

best_feasible = min(feasible_sols, key=lambda x: x.energy)
print(f"selection = {[best_feasible.sample[f'item[{i}]'] for i in range(n)]}")
print(f"sum of value = {-best_feasible.energy}")
