import numpy as np
from tqdm import tqdm
import argparse, time, pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from dataloader import DialogLoader
from model import DialogBertTransformer, MaskedNLLLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon):
    "Prepare optimizer"
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return optimizer


def configure_dataloaders(dataset, classify, batch_size):
    "Prepare dataloaders"
    if dataset == 'persuasion':
        train_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_' + classify + '_loss_mask.tsv'
        valid_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_' + classify + '_loss_mask.tsv'
        test_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_' + classify + '_loss_mask.tsv'
    else:
        train_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_loss_mask.tsv'
        valid_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_loss_mask.tsv'
        test_mask = 'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_loss_mask.tsv'
        
    train_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_' + classify + '.tsv',
        train_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_train_speakers.tsv',  
        batch_size,
        shuffle=True
    )
    
    valid_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_' + classify + '.tsv',
        valid_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_valid_speakers.tsv', 
        batch_size,
        shuffle=False
    )
    
    test_loader = DialogLoader(
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_utterances.tsv',  
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_' + classify + '.tsv',
        test_mask,
        'datasets/dialogue_level_minibatch/' + dataset + '/' + dataset + '_test_speakers.tsv', 
        batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader, test_loader



def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    assert not train or optimizer!=None
    
    if train:
        model.train()
    else:
        model.eval()
    
    for conversations, label, loss_mask, speaker_mask in tqdm(dataloader, leave=False):
        if train:
            optimizer.zero_grad()
            
        # create umask and qmask 
        lengths = [len(item) for item in conversations]
        umask = torch.zeros(len(lengths), max(lengths)).long().cuda()
        for j in range(len(lengths)):
            umask[j][:lengths[j]] = 1
            
        qmask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in speaker_mask], 
                                                batch_first=False).long().cuda()
        qmask = torch.nn.functional.one_hot(qmask)
        
        # create labels and mask
        label = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label], 
                                                batch_first=True).cuda()
        
        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask], 
                                                    batch_first=True).long().cuda()
        
        
        # obtain log probabilities
        log_prob = model(conversations, lengths, umask, qmask)
        
        # compute loss and metrics
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        labels_ = label.view(-1) 
        loss = loss_function(lp_, labels_, loss_mask)

        pred_ = torch.argmax(lp_, 1) 
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(loss_mask.view(-1).cpu().numpy())

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
    
    if dataset in ['iemocap']:
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        fscores = [avg_fscore]
        
    elif dataset in ['persuasion', 'multiwoz']:
        avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
        avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
        fscores = [avg_fscore1, avg_fscore2, avg_fscore3]
        
    elif dataset == 'dailydialog':
        if classify == 'emotion':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='weighted', labels=[0,2,3,4,5,6])*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            avg_fscore4 = round(f1_score(labels, preds, sample_weight=masks, average='micro', labels=[0,2,3,4,5,6])*100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            avg_fscore6 = round(f1_score(labels, preds, sample_weight=masks, average='macro', labels=[0,2,3,4,5,6])*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3, avg_fscore4, avg_fscore5, avg_fscore6]
            
        elif classify == 'act':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3]
    
    return avg_loss, avg_accuracy, fscores, labels, preds, masks 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weight')
    parser.add_argument('--cls-model', default='lstm', help='lstm|dialogrnn|logreg')
    parser.add_argument('--model', default='roberta', help='which model family bert|roberta|sbert; sbert is sentence transformers')
    parser.add_argument('--mode', default='0', help='which mode 0: bert or roberta base | 1: bert or roberta large; \
                                                     0, 1: bert base, large sentence transformer and 2, 3: roberta base, large sentence transformer')
    parser.add_argument('--dataset', help='which dataset iemocap|multiwoz|dailydialog|persuasion')
    parser.add_argument('--classify', help='what to classify emotion|act|intent|er|ee')
    parser.add_argument('--cattn', default='general', help='context attention for dialogrnn simple|general|general2')
    parser.add_argument('--attention', action='store_true', default=False, help='use attention on top of lstm model')
    parser.add_argument('--residual', action='store_true', default=False, help='use residual connection')
    args = parser.parse_args()

    print(args)

    global dataset
    global classify
    D_h = 200
    batch_size = args.batch_size
    n_epochs = args.epochs
    dataset = args.dataset
    classification_model = args.cls_model
    transformer_model = args.model
    transformer_mode = args.mode
    context_attention = args.cattn
    attention = args.attention
    residual = args.residual
    
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
            print ('Classifying persuador in Persuasion for Good.')
            n_classes  = 11
        elif classify == 'ee':
            print ('Classifying persuadee in Persuasion for Good.')
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
    
    model = DialogBertTransformer(D_h, classification_model, transformer_model, transformer_mode, n_classes, context_attention, attention, residual)
    
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda())
    else:
        loss_function = MaskedNLLLoss()
        
    optimizer = configure_optimizers(model, args.weight_decay, args.lr, args.adam_epsilon)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, valid_loader, test_loader = configure_dataloaders(dataset, classify, batch_size)
    
    lf = open('logs/' + dataset + '_' + transformer_model + '_mode_' + transformer_mode 
              + '_' + classification_model + '_' + classify + '.txt', 'a')
    rf = open('results/' + dataset + '_' + transformer_model + '_mode_' + transformer_mode 
              + '_' + classification_model + '_' + classify + '.txt', 'a')

    valid_losses, valid_fscores = [], []
    test_fscores = []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, train_fscore, _, _, _ = train_or_eval_model(model, loss_function,
                                                                           train_loader, e, optimizer, True)
        
        valid_loss, valid_acc, valid_fscore, _, _, _ = train_or_eval_model(model, loss_function, 
                                                                           valid_loader, e)
        
        test_loss, test_acc, test_fscore, test_label, test_pred, test_mask  = train_or_eval_model(model, loss_function,
                                                                                                  test_loader, e)
        
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_fscores.append(test_fscore)
        
        if best_loss == None or best_loss > valid_loss:
            best_loss, best_label, best_pred, best_mask =\
                    valid_loss, test_label, test_pred, test_mask
        
        x = 'Epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} valid_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        print (x)
        lf.write(x + '\n')
        
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()        
        
    print('Test performance.')
    if dataset == 'dailydialog' and classify =='emotion':  
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        score3 = test_fscores[1][np.argmin(valid_losses)]
        score4 = test_fscores[1][np.argmax(valid_fscores[1])]
        score5 = test_fscores[2][np.argmin(valid_losses)]
        score6 = test_fscores[2][np.argmax(valid_fscores[2])]
        score7 = test_fscores[3][np.argmin(valid_losses)]
        score8 = test_fscores[3][np.argmax(valid_fscores[3])]
        score9 = test_fscores[4][np.argmin(valid_losses)]
        score10 = test_fscores[4][np.argmax(valid_fscores[4])]
        score11 = test_fscores[5][np.argmin(valid_losses)]
        score12 = test_fscores[5][np.argmax(valid_fscores[5])]
        
        scores = [score1, score2, score3, score4, score5, score6, 
                  score7, score8, score9, score10, score11, score12]
        scores_val_loss = [score1, score3, score5, score7, score9, score11]
        scores_val_f1 = [score2, score4, score6, score8, score10, score12]
        
        print ('Scores: Weighted, Weighted w/o Neutral, Micro, Micro w/o Neutral, Macro, Macro w/o Neutral')
        print('F1@Best Valid Loss: {}'.format(scores_val_loss))
        print('F1@Best Valid F1: {}'.format(scores_val_f1))
        
    elif (dataset=='dailydialog' and classify=='act') or (dataset=='persuasion'):  
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        score3 = test_fscores[1][np.argmin(valid_losses)]
        score4 = test_fscores[1][np.argmax(valid_fscores[1])]
        score5 = test_fscores[2][np.argmin(valid_losses)]
        score6 = test_fscores[2][np.argmax(valid_fscores[2])]
        
        scores = [score1, score2, score3, score4, score5, score6]
        scores_val_loss = [score1, score3, score5]
        scores_val_f1 = [score2, score4, score6]
        
        print ('Scores: Weighted, Micro, Macro')
        print('F1@Best Valid Loss: {}'.format(scores_val_loss))
        print('F1@Best Valid F1: {}'.format(scores_val_f1))
        
    else:
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        scores = [score1, score2]
        print('F1@Best Valid Loss: {}; F1@Best Valid F1: {}'.format(score1, score2))
        
    scores = [str(item) for item in scores]
    
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    lf.write('\n' + str(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)) + '\n')
    lf.write(str(confusion_matrix(best_label, best_pred, sample_weight=best_mask)) + '\n')
    lf.write('-'*50 + '\n\n')
    rf.close()
    lf.close()