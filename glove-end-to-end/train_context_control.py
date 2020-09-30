from tqdm import tqdm
import os, sys
import numpy as np
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from utils import pretrained_matrix
from dataloader import DialogLoader
from model import End2EndModel, MaskedNLLLoss
from torchnlp.encoders.text import SpacyEncoder
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

def configure_dataloaders(dataset, classify, batch_size):
    "Prepare dataloaders"
    
    if dataset == 'persuasion':
        utt_file1 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_train_' + classify + '_utterances.tsv'
        utt_file2 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_valid_' + classify + '_utterances.tsv'
        utt_file3 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_test_' + classify + '_utterances.tsv'
        mask_file1 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_train_' + classify + '_loss_mask.tsv'
        mask_file2 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_valid_' + classify + '_loss_mask.tsv'
        mask_file3 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_test_' + classify + '_loss_mask.tsv'
    else:
        utt_file1 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_train_utterances.tsv'
        utt_file2 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_valid_utterances.tsv'
        utt_file3 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_test_utterances.tsv'
        mask_file1 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_train_loss_mask.tsv'
        mask_file2 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_valid_loss_mask.tsv'
        mask_file3 = 'datasets/utterance_level/' + dataset + '/' + dataset + '_test_loss_mask.tsv'
        
    
    train_loader = DialogLoader(
        utt_file1,  
        'datasets/utterance_level/' + dataset + '/' + dataset + '_train_' + classify + '.tsv',
        mask_file1,
        mask_file1, # dummy speaker mask
        batch_size,
        shuffle=True
    )
    
    valid_loader = DialogLoader(
        utt_file2,  
        'datasets/utterance_level/' + dataset + '/' + dataset + '_valid_' + classify + '.tsv',
        mask_file2, 
        mask_file2, # dummy speaker mask
        batch_size,
        shuffle=False
    )
    
    test_loader = DialogLoader(
        utt_file3,  
        'datasets/utterance_level/' + dataset + '/' + dataset + '_test_' + classify + '.tsv',
        mask_file3, 
        mask_file3, # dummy speaker mask
        batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader, test_loader

def update_context(conversations, target_idx, cc):
    return [
            (conv[: max(0, idx.item() - cc['past_context'])] if cc['past_del'] else conv[max(0, idx.item() - cc['past_context']) : idx.item()]) +
            [conv[idx.item()]] +
            (conv[idx.item() + 1 + cc['future_context'] :] if cc['future_del'] else conv[idx.item() + 1 : idx.item() + 1 + cc['future_context']])
            for conv, idx in zip(conversations, target_idx)
            ]

def train_or_eval_model(model, loss_function, dataloader, epoch, cc=None, optimizer=None, train=False, one_element=False):
    losses, preds, labels, masks = [], [], [], []
    assert not train or optimizer!=None

    if train:
        model.train()
    else:
        model.eval()

    for conversations, label, loss_mask, dummy_speaker_mask, dummy_indices in tqdm(dataloader, leave=False):
        if train:
            optimizer.zero_grad()

        # create labels and mask
        if cc:
            loss_mask_ = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask],
                                                        batch_first=True).cuda()
            target_idx = loss_mask_.argmax(dim=1)
            conversations = update_context(conversations, target_idx, cc)
            label         = update_context(label, target_idx, cc)
            loss_mask     = update_context(loss_mask, target_idx, cc)

        label = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label],
                                                batch_first=True).cuda()
        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask],
                                                    batch_first=True).cuda()

        # create umask and qmask
        lengths = [len(item) for item in conversations]
        umask = torch.zeros(len(lengths), max(lengths)).long().cuda()
        for j in range(len(lengths)):
            umask[j][:lengths[j]] = 1

        # obtain log probabilities
        log_prob = model(conversations, lengths, umask)

        if dataset == 'persuasion' and classify == 'er':
            log_prob = log_prob[0]
        if dataset == 'persuasion' and classify == 'ee':
            log_prob = log_prob[1]

        # compute loss and metrics
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1)
        loss = loss_function(lp_, labels_, loss_mask)

        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(loss_mask.view(-1).cpu().numpy())

        # save_grad = True

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)

    if dataset in ['iemocap', 'multiwoz']:
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        fscores = [avg_fscore]
        if one_element:
            fscores = fscores[0]

    elif dataset in ['persuasion']:
        avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
        avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
        fscores = [avg_fscore1, avg_fscore2, avg_fscore3]
        if one_element:
            fscores = fscores[2]

    elif dataset == 'dailydialog':
        if classify == 'emotion':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='weighted', labels=[0,2,3,4,5,6])*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            avg_fscore4 = round(f1_score(labels, preds, sample_weight=masks, average='micro', labels=[0,2,3,4,5,6])*100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            avg_fscore6 = round(f1_score(labels, preds, sample_weight=masks, average='macro', labels=[0,2,3,4,5,6])*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3, avg_fscore4, avg_fscore5, avg_fscore6]
            if one_element:
                fscores = fscores[5]

        elif classify == 'act':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3]
            if one_element:
                fscores = fscores[2]

    return avg_loss, avg_accuracy, fscores, labels, preds, masks

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout probability.")
    parser.add_argument('--rec-dropout', default=0.1, type=float, help="DialogRNN Dropout probability.")
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=10, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm model')
    parser.add_argument('--cls-model', default='lstm', help='lstm or logreg')
    parser.add_argument('--mode', default='840B', help='which glove model')
    parser.add_argument('--dataset', default='iemocap', help='which dataset')
    parser.add_argument('--classify', help='what to classify')
    parser.add_argument('--cattn', default='general', help='context attention for dialogrnn simple|general|general2')
    parser.add_argument('--run' , help='which run')

    parser.add_argument('--cc-active', action='store_true', default=False, help='cc active')
    parser.add_argument('--cc-past-context', type=int, default=-1, help='# past utterances')
    parser.add_argument('--cc-future-context', type=int, default=-1, help='# future utterances')
    parser.add_argument('--cc-past-del', action='store_true', default=False, help='remove context')
    parser.add_argument('--cc-future-del', action='store_true', default=False, help='remove context')
    parser.add_argument('--cc-train', action='store_true', default=False, help='cc on training set')
    parser.add_argument('--cc-dev', action='store_true', default=False, help='cc on dev set')
    parser.add_argument('--cc-test', action='store_true', default=False, help='cc on test set')

    parser.add_argument('--inference', default=None, help='model ID')

    args = parser.parse_args()

    print(args)

    if not args.inference:
        run_ID = int(time.time())
        print(f'model_ID: {run_ID}')

    global dataset
    global classify
    dataset = args.dataset
    cc = {
            'active': args.cc_active,
            'past_context': 0 if args.cc_past_context<0 else args.cc_past_context,
            'future_context': 0 if args.cc_future_context<0 else args.cc_future_context,
            'past_del': True if args.cc_past_context<0 else args.cc_past_del,
            'future_del': True if args.cc_future_context<0 else args.cc_future_del,
            'train': args.cc_train,
            'dev': args.cc_dev,
            'test': args.cc_test,
            }
    D_h = 100
    D_e = 100
    if dataset in ['multiwoz']:
        D_e = 200
    cnn_output_size = 100
    cnn_filters = 100
    cnn_kernel_sizes = (1,2,3)
    if dataset in ['multiwoz']:
        cnn_kernel_sizes = (2,3,4)
    mode = args.mode
    cnn_dropout = args.dropout
    dropout = args.dropout
    rec_dropout = args.rec_dropout
    attention = args.attention
    batch_size = args.batch_size
    n_epochs = args.epochs
    classification_model = args.cls_model
    context_attention = args.cattn

    if dataset == 'iemocap':
        print ('Classifying emotion in iemocap.')
        classify = 'emotion'
        n_classes  = 6
        loss_weights = torch.FloatTensor([1.0, 0.60072, 0.38066, 0.54019, 0.67924, 0.34332])

    elif dataset == 'multiwoz':
        print ('Classifying intent in multiwoz.')
        classify = 'intent'
        n_classes  = 11

    elif dataset == 'persuasion':
        classify = args.classify
        if classify == 'er':
            print ('Classifying persuador in Persuasion for good.')
            n_classes  = 11
        elif classify == 'ee':
            print ('Classifying persuadee in Persuasion for good.')
            n_classes  = 13
        else:
            raise ValueError('--classify must be er or ee for persuasion')

    elif dataset == 'dailydialog':
        classify = args.classify
        if classify == 'emotion':
            print ('Classifying emotion in dailydialog.')
            n_classes  = 7
        elif classify == 'act':
            print ('Classifying act in dailydialog.')
            n_classes  = 4
        else:
            raise ValueError('--classify must be emotion or act for dailydialog')

    train_loader, valid_loader, test_loader = configure_dataloaders(dataset, classify, batch_size)

    ## Tokenizer and Embedding Matrix
    if os.path.isfile('datasets/' + dataset + '/' + dataset + mode + '_embedding.matrix'):
        tokenizer = pickle.load(open('datasets/' + dataset + '/' + dataset  + mode + '.tokenizer', 'rb'))
        embedding_matrix = pickle.load(open('datasets/' + dataset + '/' + dataset + mode + '_embedding.matrix', 'rb'))
        print ('Tokenizer and embedding matrix exists. Loaded from pickle files.')
    else:
        print ('Creating tokenizer and embedding matrix.')
        all_utterances = []
        for loader in [train_loader, valid_loader, test_loader]:
            for conversations, label, loss_mask, speakers, indices in loader:
                all_utterances += [sent.lower() for conv in conversations for sent in conv]

        tokenizer = SpacyEncoder(all_utterances)
        id_to_token = {i: item for i, item in enumerate(tokenizer.vocab)}

        if mode == '6B':
            embedding_matrix = pretrained_matrix('glove/glove.6B.300d.txt', id_to_token)
        elif mode == '840B':
            embedding_matrix = pretrained_matrix('glove/glove.840B.300d.txt', id_to_token)

        pickle.dump(tokenizer, open('datasets/' + dataset + '/' + dataset + mode + '.tokenizer', 'wb'))
        pickle.dump(embedding_matrix, open('datasets/' + dataset + '/' + dataset + mode + '_embedding.matrix', 'wb'))
        print ('Done.')

    vocab_size, embedding_dim = embedding_matrix.shape

    model = End2EndModel(dataset, vocab_size, embedding_dim, tokenizer, classification_model,
                         cnn_output_size, cnn_filters, cnn_kernel_sizes, cnn_dropout,
                         D_e, D_h, n_classes, dropout, attention, context_attention, rec_dropout)

    if args.inference:
        if dataset == 'iemocap':
            model.load_state_dict(torch.load(f'saved/iemocap/lstm_emotion_{args.inference}.pt'))

        elif dataset == 'multiwoz':
            model.load_state_dict(torch.load(f'saved/multiwoz/lstm_intent_{args.inference}.pt'))

        elif dataset == 'persuasion' and classify == 'er':
            model.load_state_dict(torch.load(f'saved/persuasion/lstm_er_{args.inference}.pt'))

        elif dataset == 'persuasion' and classify == 'ee':
            model.load_state_dict(torch.load(f'saved/persuasion/lstm_ee_{args.inference}.pt'))

        elif dataset == 'dailydialog' and classify == 'emotion':
            model.load_state_dict(torch.load(f'saved/dailydialog/lstm_emotion_{args.inference}.pt'))

        elif dataset == 'dailydialog' and classify == 'act':
            model.load_state_dict(torch.load(f'saved/dailydialog/lstm_act_{args.inference}.pt'))

        n_epochs = 1

    model.init_pretrained_embeddings(embedding_matrix)
    model.cuda()

    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda())
    else:
        loss_function = MaskedNLLLoss()

    if not args.inference:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    valid_losses, valid_fscores = [], []
    test_fscores = []
    best_loss, best_label, best_pred, best_mask, best_fscore = None, None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        
        if not args.inference:
            train_loss, train_acc, train_fscore, _, _, _ = train_or_eval_model(model, loss_function,
                                                                           train_loader, e,
                                                                           cc if cc['train'] and cc['active'] else None,
                                                                           optimizer=optimizer if not args.inference else None,
                                                                           train=True if not args.inference else False,
                                                                           one_element=True)

            valid_loss, valid_acc, valid_fscore, _, _, _ = train_or_eval_model(model, loss_function,
                                                                           valid_loader, e,
                                                                           cc if cc['dev'] and cc['active'] else None,
                                                                           one_element=True
                                                                           )


            valid_losses.append(valid_loss)
            valid_fscores.append(valid_fscore)
        
        test_loss, test_acc, test_fscore, test_label, test_pred, test_mask  = train_or_eval_model(model, loss_function,
                                                                                                  test_loader, e,
                                                                                                  cc if cc['test'] and cc['active'] else None, one_element=True)
        test_fscores.append(test_fscore)

        # WARNING: model hyper-parameters are not stored
        if not args.inference:
            if best_fscore == None or valid_fscore > best_fscore:
                best_fscore = valid_fscore
                if not os.path.exists('mapping/'):
                    os.makedirs('mapping/')
                with open(f'mapping/{dataset}_classify_{classify}_run{args.run}_{run_ID}.tsv', 'w') as f:
                    f.write(f'{args}\t{run_ID}\t{best_fscore}')

                if dataset == 'iemocap':
                    dirName = 'saved/iemocap/'
                    if not os.path.exists(dirName):
                        os.makedirs(dirName)
                    torch.save(model.state_dict(), f'saved/iemocap/lstm_emotion_run{args.run}_{run_ID}.pt')

                elif dataset == 'multiwoz':
                    dirName = 'saved/multiwoz/'
                    if not os.path.exists(dirName):
                        os.makedirs(dirName)
                    torch.save(model.state_dict(), f'saved/multiwoz/lstm_intent_run{args.run}_{run_ID}.pt')

                elif dataset == 'persuasion' and classify == 'er':
                    dirName = 'saved/persuasion/'
                    if not os.path.exists(dirName):
                        os.makedirs(dirName)
                    torch.save(model.state_dict(), f'saved/persuasion/lstm_er_run{args.run}_{run_ID}.pt')

                elif dataset == 'persuasion' and classify == 'ee':
                    dirName = 'saved/persuasion/'
                    if not os.path.exists(dirName):
                        os.makedirs(dirName)
                    torch.save(model.state_dict(), f'saved/persuasion/lstm_ee_run{args.run}_{run_ID}.pt')

                elif dataset == 'dailydialog' and classify == 'emotion':
                    dirName = 'saved/dailydialog/'
                    if not os.path.exists(dirName):
                        os.makedirs(dirName)
                    torch.save(model.state_dict(), f'saved/dailydialog/lstm_emotion_run{args.run}_{run_ID}.pt')

                elif dataset == 'dailydialog' and classify == 'act':
                    dirName = 'saved/dailydialog/'
                    if not os.path.exists(dirName):
                        os.makedirs(dirName)
                    torch.save(model.state_dict(), f'saved/dailydialog/lstm_act_run{args.run}_{run_ID}.pt')

        if not args.inference:
            if best_loss == None or best_loss > valid_loss:
                best_loss, best_label, best_pred, best_mask =\
                    valid_loss, test_label, test_pred, test_mask
            x = 'Epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} valid_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
            print (x)

    # valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()

    if not args.inference:
        sys.exit(0)
        
    else:
        print (test_fscores)
        ccf = open('results/context_control/' + dataset + '_glove_utt_level_context_control_' + classification_model + '_' + classify + '.txt', 'a')
        ccf.write(str(test_fscores[0]) + '\t' + str(args.inference) + '\t' + str(args) + '\n')
        
