import torch
from ordered_set import OrderedSet
from model.models import *
ent_set, rel_set = OrderedSet(), OrderedSet()
for split in ['train', 'test', 'valid']:
    for line in open('./data/{}/{}.txt'.format('medical', split)):
                # list =  line.strip().split('\t')
                # if len(list)!=3:
                #     print(line)
                #     print(list)
        sub, rel, obj = map(str.lower, line.strip().split('\t'))
        ent_set.add(sub)
        rel_set.add(rel)
        ent_set.add(obj)

ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
#rel2id.update({rel + '_reverse': idx + len(rel2id) for idx, rel in enumerate(rel_set)})

id2ent = {idx: ent for ent, idx in ent2id.items()}
id2rel = {idx: rel for rel, idx in rel2id.items()}
load_path='./checkpoints/test_medical_2_09_05_2023_12:52:39'
state = torch.load(load_path)
p= argparse.Namespace(**state['args'])
print("---------")
#print(state)
print("---------")
#print(state['edge_index'])
model=RagatInteractE(state['edge_index'], state['edge_type'], params=p)
state_dict = state['state_dict']
model.load_state_dict(state_dict)
model.eval()
head=ent2id['巴瑞特综合征']
rel=rel2id['宜吃']
topk=10
with torch.no_grad():
    device = torch.device('cuda')
    model.to(device)
    all_ents = torch.arange(p.num_ent, device=device)
    rel_vec = torch.full((p.num_ent,), rel, dtype=torch.long, device=device)
    sub = torch.full((p.num_ent,), head, dtype=torch.long, device=device)
    obj = all_ents
    batch_size = 1024
    num_batches = (p.num_ent + batch_size - 1) // batch_size
    scores_list = []
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, p.num_ent)
        sub_batch = sub[start:end]
        rel_batch = rel_vec[start:end]
        obj_batch = all_ents[start:end]
        scores_batch = model(sub_batch, rel_batch, obj_batch)
        scores_list.append(scores_batch)
    scores = torch.cat(scores_list, dim=0)
    ranks = torch.argsort(torch.argsort(-scores))
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    top_tails = sorted_indices[:10]
    top_scores = sorted_scores[:10]
    results = {}
    results['tails'] = top_tails.tolist()
    results['scores'] = top_scores.tolist()
print(len(results['tails'][0]))
print(len(ent_set))
# for i in results['tails'][0]:
#     print(id2ent[i])
for i in range(20):
    print(id2ent[results['tails'][0][i]]+" : "+str(results['scores'][0][i]))