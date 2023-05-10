from ordered_set import OrderedSet
import torch

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
for item in id2rel.items():
    print(item)

def predict(self, head, rel,  topk=10):
    """
    Function to run model evaluation for a given mode

    Parameters
    ----------
    head: (int)        The input head entity ID
    rel: (int)         The input relation ID
    split: (string)    If split == 'valid' then evaluate on the validation set, else the test set
    mode: (string)     Can be 'head_batch' or 'tail_batch'
    topk: (int)        Number of top entities to return

    Returns
    -------
    results:            A list of the top-k tail entities with their corresponding scores
    """
    self.model.eval()

    with torch.no_grad():
        results = {}
        device = torch.device('cuda')
        pred = self.model.forward(torch.tensor(head).to(device), torch.tensor(rel).to(device))
        target_pred = pred.squeeze()
        b_range = torch.arange(target_pred.size()[0], device=device)
        
        # filter setting
        pred = torch.where(target_pred.byte(), -torch.ones_like(pred) * 10000000, pred)
        pred[b_range, target_pred] = target_pred
        
        sorted_scores, sorted_indices = torch.topk(pred, k=topk, dim=1)

        results = [(ind.item(), score.item()) for ind, score in zip(sorted_indices[0], sorted_scores[0])]
            
    return results
load_path='./checkpoints/test_medical_2_08_05_2023_16:54:39'
state = torch.load(load_path)
state_dict = state['state_dict']
best_val = state['best_val']
best_val_mrr = best_val['mrr']
best_hit10=best_val['hits@10']
args=state['args']
model = RagatInteractE(self.edge_index, self.edge_type, params=args)
self.model.load_state_dict(state_dict)
self.optimizer.load_state_dict(state['optimizer'])