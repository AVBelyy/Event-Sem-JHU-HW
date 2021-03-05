import collections
import itertools
import json
import pdb
import os

from decomp import UDSCorpus


roles_preds = {
    'agent': lambda e: (e['volition']['value'] > 0 or e['instigation']['value'] > 0) and e['existed_before']['value'] > 0,
    'patient': lambda e: e['volition']['value'] < 0 and e['instigation']['value'] < 0 and e['change_of_state']['value'] > 0,
    'theme': lambda e: e['volition']['value'] < 0 and e['instigation']['value'] < 0 and e['change_of_state']['value'] < 0,
    'instrument': lambda e: e['was_used']['value'] > 0 and e['existed_during']['value'] > 0 and e['volition']['value'] < 0,
    'beneficiary': lambda e: e['instigation']['value'] < 0 and e['was_for_benefit']['value'] > 0,
}


def parse_edge(edge):
    pred_head_idx = None
    arg_heax_idx = None

    for part in edge:
        part_type, part_idx = part.split('-')[-2:]
        part_idx = int(part_idx) - 1  # convert 1-based indices to 0-based
        assert part_type in ('pred', 'arg')
        if part_type == 'pred':
            pred_head_idx = part_idx
        else:
            arg_heax_idx = part_idx
    
    assert pred_head_idx is not None
    assert arg_heax_idx is not None
    
    return pred_head_idx, arg_heax_idx


def create_data_split(split_name, out_dir_path):
    items_sents = collections.OrderedDict()
    items_labels = collections.OrderedDict()
    
    # Read annotated graphs from the split
    agent_and_patient_cnt = 0
    uds_corpus = UDSCorpus(split=split_name, version='1.0')
    for graph_id, graph in uds_corpus.items():
        sent_tokens = list(graph.sentence.split())
        for edge, props in graph.semantics_edges().items():
            if 'protoroles' in props:
                pred_head_idx, arg_head_idx = parse_edge(edge)
                item_id = (graph_id, pred_head_idx, arg_head_idx)
                items_labels[item_id] = {}
                items_sents[item_id] = sent_tokens
                for role, pred in roles_preds.items():
                    try:
                        role_applies = pred(props['protoroles'])
                        items_labels[item_id][role] = role_applies
                    except:
                        pass

    # Write output JSONL files
    for role in roles_preds.keys():
        out_role_dir_path = f'{out_dir_path}/{role}'
        out_role_file_path = f'{out_role_dir_path}/{split_name}.jsonl'
        if not os.path.exists(out_role_dir_path):
            os.makedirs(out_role_dir_path)
        with open(out_role_file_path, 'w') as fout:
            for item_id, item_labels in items_labels.items():
                if role in item_labels:
                    graph_id, pred_head_idx, arg_head_idx = item_id
                    out_item = {
                        'graph_id': graph_id,
                        'pred_head_idx': pred_head_idx,
                        'arg_head_idx': arg_head_idx,
                        'sent': items_sents[item_id],
                        'label': item_labels[role],
                    }
                    fout.write(f'{json.dumps(out_item)}\n')
    
    # Print label stats
    print('split:', split_name)
    for role in roles_preds.keys():
        label_cnt = collections.Counter()
        for item_labels in items_labels.values():
            if role in item_labels:
                label_cnt[item_labels[role]] += 1
        print(f'  {role}: {label_cnt}')
    for role1, role2 in itertools.combinations(roles_preds.keys(), 2):
        label_cnt = collections.Counter()
        for item_labels in items_labels.values():
            if role1 in item_labels and role2 in item_labels:
                is_both = item_labels[role1] and item_labels[role2]
                label_cnt[is_both] += 1
        print(f'  {role1} & {role2}: {label_cnt}')


if __name__ == '__main__':
    for split_name in ('dev', 'test', 'train'):
        create_data_split(split_name, 'data/')  # TODO: make external param
