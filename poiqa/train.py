import argparse
import torch
import numpy as np
import json
import os
import sys
import subprocess
import logging
from poiqa import utils, data, networkwrapper, config
import configparser

logger = logging.getLogger()

DATA_DIR='../data/working/train'
EMBED_DIR='../data/embed/'
MODEL_DIR='../data/output'

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    parser.register('type', 'bool', str2bool)
    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data_workers', type=int, default=2,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random_seed', type=int, default=1111,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num_epochs', type=int, default=10,
                         help='Train data iterations')
    runtime.add_argument('--batch_size', type=int, default=16,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=16,
                         help='Batch size during validation/testing')
    runtime.add_argument('--mode', type=str, default='train', help='(train,test)')
    runtime.add_argument('--train_ratio',type=float,default=0.8)

    files = parser.add_argument_group('Input and output files')
    files.add_argument('--train_file', type=str, default='q-sample.txt',
                       help='loading the tweet json file.')
    files.add_argument('--kb_file', type=str, default='kb-sample.txt',
                       help='loading the poi data json file.')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       help='Space-separated pretrained embeddings file')

    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--restrict_vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=True,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand_dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # General
    general = parser.add_argument_group('General')
    general.add_argument('--valid-metric', type=str, default='negdis',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=10,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=False,
                         help='Sort batches by length for speed')

    return parser


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.kb_file = os.path.join(args.data_dir, args.kb_file)
    if not os.path.isfile(args.kb_file):
        raise IOError('No such file: %s' % args.kb_file)

    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained)

    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    if args.network in ['mem', 'mem-attn', 'mem-bow']:
        args.use_kb = True

    if args.mode == 'test':
        args.train_ratio=0
    return args


# -----------------------------
# initialization from scratch.
# -----------------------------

def init_from_scratch(args, examples, kb):
    """New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    if kb:
        feature_dict = utils.build_feature_dict(args, examples + kb)
    else:
        feature_dict = utils.build_feature_dict(args, examples)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    # Build a dictionary from the data questions + kb words
    logger.info('-' * 100)
    logger.info('Build dictionary')
    if kb:
        word_dict = utils.build_word_dict(args, examples + kb)
    else:
        word_dict = utils.build_word_dict(args, examples)
    logger.info('Num words = %d' % len(word_dict))

    # Initialize model
    model = networkwrapper.NetworkWrapper(config.get_model_args(args), word_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        batch_size = ex[0].shape[0]
        loss = model.update(ex, 'train')
        train_loss.update(loss[0], batch_size)

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.4f| elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


def validate(args, data_loader, model):
    validate_loss = utils.AverageMeter()
    validate_acc1 = utils.AverageMeter()
    validate_acc5 = utils.AverageMeter()
    validate_dis = utils.AverageMeter()
    epoch_time = utils.Timer()
    # Run one epoch
    for idx, ex in enumerate(data_loader):
        batch_size = ex[0].shape[0]
        loss, dis, acc, entropy = model.update(ex, 'test')
        dis_sum = torch.sum(dis)
        acc1_sum = torch.sum(acc[0])
        acc5_sum = torch.sum(acc[1])
        validate_loss.update(loss[0], batch_size)  # convert from tensor to float
        validate_acc1.update(acc1_sum, batch_size)
        validate_acc5.update(acc5_sum,batch_size)
        validate_dis.update(dis_sum, batch_size)

    logger.info('validation done. Time for epoch = %.2f (s)' % epoch_time.time() +
                'loss = %.4f, acc@1 = %.4f,acc@5= %.4f dis=%.2f' % (validate_loss.avg,
                                                       validate_acc1.avg,
                                                       validate_acc5.avg,
                                                       validate_dis.avg))
    return {
        'negloss': -validate_loss.avg,
        'acc@1': validate_acc1.avg,
        'acc@5':validate_acc5.avg,
        'dis': validate_dis.avg,
        'negdis':-validate_dis.avg,
    }

def test(args,data_loader,model):
    raw_dis = []
    loss = utils.AverageMeter()
    acc1 = utils.AverageMeter()
    acc5 = utils.AverageMeter()
    dis = utils.AverageMeter()
    epoch_time = utils.Timer()
    # Run one epoch
    for idx, ex in enumerate(data_loader):
        batch_size = ex[0].shape[0]
        loss, dis, acc, entropy = model.update(ex, 'test')
        loss.update(loss[0], batch_size)  # convert from tensor to float
        acc1.update(torch.sum(acc[0]), batch_size)
        acc5.update(torch.sum(acc[1]), batch_size)
        dis.update(torch.sum(dis), batch_size)
        raw_dis.append(dis)

    dis_tensor = torch.cat(raw_dis).cpu()
    med = torch.median(dis_tensor)
    distance = dis_tensor.numpy()
    filename = args.model_dir+'/'+ args.model_name + 'prediction.npy'

    np.save(filename,distance)
    logger.info('validation done. Time for epoch = %.2f (s)' % epoch_time.time() +
                'loss = %.4f, acc@1 = %.4f,acc@5= %.4f mean dis=%.2f median dis=%.2f' % (loss.avg,
                                                                    acc1.avg,
                                                                    acc5.avg,
                                                                    dis.avg,med))
    return {
        'negloss': -loss.avg,
        'acc@1': acc1.avg,
        'acc@5': acc5.avg,
        'dis': dis.avg,
        'negdis': -dis.avg,
    }



# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')

    train_exs = utils.load_data(args, args.train_file)
    if args.use_kb:
        kb = utils.load_data(args, args.kb_file)
    # list of dict



    logger.info('Num train examples = %d' % len(train_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = networkwrapper.NetworkWrapper.load_checkpoint(checkpoint_file, args)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = networkwrapper.NetworkWrapper.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = utils.load_words(args, train_exs + kb)
                added = model.expand_dictionary(words)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)

        else:
            logger.info('Training model from scratch...')

            if args.use_kb:
                model = init_from_scratch(args, train_exs, kb)
                model.init_kb(kb)
            else:
                model = init_from_scratch(args, train_exs, None)
        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATOR
    logger.info('-' * 100)
    logger.info('Make data loaders')

    dataset = data.GeoTweetDataset(train_exs, model)
    [training_sampler, valid_sampler] = dataset.TrainValidationSplit(
        ratio=args.train_ratio,
        shuffle=True,
        seed=args.random_seed)
    if args.train_ratio==0:
        logger.info('No training')
    if args.mode == 'train' and args.train_ratio!=0:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.data_workers,
                                                   collate_fn=data.qa_collate,
                                                   sampler=training_sampler,
                                                   pin_memory=args.cuda)

    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.data_workers,
                                               collate_fn=data.qa_collate,
                                               sampler=valid_sampler,
                                               pin_memory=args.cuda)

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    if args.mode == 'train':
        stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': -np.inf}
        for epoch in range(start_epoch, args.num_epochs):
            stats['epoch'] = epoch

            # Train

            train(args, train_loader, model, stats)

            result = validate(args, valid_loader, model)
            # Save best valid
            # distance metric the smaller the better
            if result[args.valid_metric] > stats['best_valid']:
                logger.info('Best valid: %s = %.4f (epoch %d, %d updates)' %
                            (args.valid_metric, result[args.valid_metric],
                             stats['epoch'], model.updates))
                model.save(args.model_file)
                stats['best_valid'] = result[args.valid_metric]
    elif args.mode == 'test':
        test(args, valid_loader, model)

    return


if __name__ == '__main__':
    # Parse cmdline args and setup environment

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    # general settings
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
