import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from tqdm import tqdm

from dataset import Synth90kDataset, synth90k_collate_fn
from model import CRNN
from ctc_decoder import ctc_decode
from config import evaluate_config as config

torch.backends.cudnn.enabled = False

def avg_cer(preds:list, refs:list):
    cer_sum = 0
    for pred, ref in zip(preds, refs):
        len_pred, len_ref = len(pred), len(ref)
        cost_mat = np.zeros((len_ref+1, len_pred+1), dtype=np.int16) # dp table
        for i in range(len_ref+1):
            cost_mat[i][0] = i
        for j in range(len_pred+1):
            cost_mat[0][j] = j

        for i in range(1, len_ref+1):
            for j in range(1, len_pred+1):
                if ref[i-1] == pred[j-1]:
                    cost_mat[i][j] = cost_mat[i-1][j-1]
                else:
                    cost_mat[i][j] = min(cost_mat[i-1][j]+1,\
                                         cost_mat[i-1][j-1]+1,\
                                         cost_mat[i][j-1]+1)
        cer_sum += cost_mat[len_ref][len_pred] / len_ref
    return cer_sum / len(preds)

def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    crnn.eval()

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    tot_cer = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            images, targets, target_lengths = [d.to(device) for d in data]

            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_count += batch_size
            tot_loss += loss.item()
            
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                word_cer = avg_cer([pred], [real])
                tot_cer += word_cer
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred, word_cer))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases,
        'avg_cer' : tot_cer / (tot_count / batch_size)
    }
    return evaluation

old_test_dataset = None
def main(config=config, reuse_dataset=False):
    global old_test_dataset
    eval_batch_size = config['eval_batch_size']
    cpu_workers = config['cpu_workers']
    reload_checkpoint = config['reload_checkpoint']

    img_height = config['img_height']
    img_width = config['img_width']
    img_limit = config['img_limit']
    use_mem_buffer = config['use_mem_buffer']
    start_sample_index = config['start_sample_index']
    image_preprocess = config['image_preprocess']
    dump_dest = config['dump_dest']
    dump_name_suffix = config['dump_name_suffix']
    save_pkl = config['save_pkl']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if not old_test_dataset:
        old_test_dataset = Synth90kDataset(root_dir=config['data_dir'], 
                                           mode='test',
                                           img_height=img_height, 
                                           img_width=img_width,
                                           img_limit=img_limit,
                                           use_mem_buffer=use_mem_buffer,
                                           start_sample_index=start_sample_index,
                                           image_preprocess=image_preprocess,
                                           dump_dest=dump_dest,
                                           dump_name_suffix=dump_name_suffix,
                                           save_pkl=save_pkl)
    if reuse_dataset:
        test_dataset = old_test_dataset
    else:
        test_dataset = Synth90kDataset(root_dir=config['data_dir'], 
                                           mode='test',
                                           img_height=img_height, 
                                           img_width=img_width,
                                           img_limit=img_limit,
                                           use_mem_buffer=use_mem_buffer,
                                           start_sample_index=start_sample_index,
                                           image_preprocess=image_preprocess,
                                           dump_dest=dump_dest,
                                           dump_name_suffix=dump_name_suffix,
                                           save_pkl=save_pkl)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cpu_workers,
        collate_fn=synth90k_collate_fn)

    num_class = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    evaluation = evaluate(crnn, test_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))
    return evaluation


if __name__ == '__main__':
    main()
