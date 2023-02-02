import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pyqubo import Array, Constraint, Placeholder
from pprint import pprint
import neal
from dwave.system.samplers import DWaveSampler
from minorminer.busclique import find_clique_embedding
from dwave.system.composites import FixedEmbeddingComposite
import dwave_networkx as dnx
import time

df = pd.read_csv('iot.txt', sep=',')
#print(df.head())

dfn=df[df["fail"] == 0]
dfs=df[df["fail"] == 1]

dfnr= dfn[['selfLR','ClinLR','DoleLR','PID']]
dfsr= dfs[['selfLR','ClinLR','DoleLR','PID']]
dfr = df[['selfLR','ClinLR','DoleLR','PID','fail']]
dfrr= dfr.head(170)
dfrrx = dfrr.drop('fail', axis = 1)

dfrrx['Average Score'] = dfrrx.mean(axis=1)-5
print(dfrrx)

x = Array.create('x',shape=len(dfrrx.index), vartype='BINARY')
h_cost = 0
N = len(x)
for i in range(N):
  for j in range(N):
    h_cost +=x[i]*(dfrrx['Average Score'].iloc[i])*x[j]*(dfrrx['Average Score'].iloc[i])-x[i]+2*x[i]*x[j]


modelh =h_cost.compile()
qubo, offset = modelh.to_qubo()

#simulated annelaing
sampler = neal.SimulatedAnnealingSampler()
bqm = modelh.to_bqm()
start = time.perf_counter()
sampleset = sampler.sample(bqm, num_reads=10)
decoded_samples = modelh.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)

end = time.perf_counter() 
ms = (end-start) * 10**6
print(f"Elapsed {ms:.03f} micro secs.")
pprint(best_sample.sample) 

dw_sampler = DWaveSampler(endpoint="https://na-west-1.cloud.dwavesys.com/sapi/v2",token="DEV-a03a6447bd1ad04cd8d5b9102cc9888812bde882",solver="Advantage_system6.1")
graph_size=16
sampler_size=len(modelh.variables)
p16_working_graph = dnx.pegasus_graph(graph_size,node_list=dw_sampler.nodelist,edge_list=dw_sampler.edgelist)
embedding = find_clique_embedding(sampler_size,p16_working_graph)
sampler = FixedEmbeddingComposite(dw_sampler, embedding)
sampler_kwargs = {"num_reads": 100,"annealing_time": 20,"num_spin_reversal_transforms": 4,"auto_scale": True,"chain_strength": 2.0,"chain_break_fraction": True}


bqm = modelh.to_bqm(index_label=True)
bqm.normalize()
start = time.perf_counter()
sampleset = sampler.sample(bqm, **sampler_kwargs)
#dec_samples = model.decode_sampleset(sampleset,feed_dict=feed_dict)
decoded_samples = modelh.decode_sampleset(sampleset)

#bqm = modelh.to_bqm()
#sampleset = sampler.sample(bqm, num_reads=10)
#decoded_samples = modelh.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)
# record end time
end = time.perf_counter()
# find elapsed time in seconds
ms = (end-start) * 10**6
print(f"Elapsed {ms:.03f} micro secs.")
pprint(best_sample.sample) 