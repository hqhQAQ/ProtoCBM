import torch
import logging
import torch.nn.functional as F
import util.utils as utils
from util.local_parts import train_pos_weights
from util.rotate_tensor import multiple_rotate_all, mask_tensor


def _train_or_test(model, epoch, dataloader, tb_writer, iteration, optimizer=None, use_l1_mask=True,
                   coefs=None, args=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    n_examples = 0
    n_correct = 0
    n_batches = 0

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 60
    pos_weight = torch.from_numpy(train_pos_weights).cuda()

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0

    for data_item in metric_logger.log_every(dataloader, print_freq, header):
        if len(data_item) == 2:
            image, label = data_item
        else:
            image, label, attributes = data_item
            attributes = torch.stack(attributes).permute(1, 0).type(torch.FloatTensor).cuda()

        attributes_criterion = torch.nn.BCEWithLogitsLoss()
        # attributes_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        input = image.cuda()
        target = label.cuda()
        bz = target.shape[0]

        # Augment Images
        if epoch < args.proto_epochs and args.use_mse_loss:
            input = multiple_rotate_all(input, all_rotate_times=[0, 1])

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, (_, proto_acts, shallow_feas, deep_feas, all_feas) = model(input)
            if isinstance(output, tuple):
                logits, logits_attri, attributes_logits = output
            # Select the top output
            logits, logits_attri, attributes_logits, proto_acts, shallow_feas, deep_feas = \
                logits[:bz], logits_attri[:bz], attributes_logits[:bz], proto_acts[:bz], shallow_feas[:bz], deep_feas[:bz]

            del input
            # Compute loss
            attributes_cost = attributes_criterion(attributes_logits, attributes)
            logits = logits if epoch < args.proto_epochs else logits_attri
            cross_entropy = torch.nn.functional.cross_entropy(logits, target)

            model_without_ddp = model.module if hasattr(model, 'module') else model
            if epoch < args.proto_epochs:
                ortho_cost = model_without_ddp.get_ortho_loss()
                consis_cost = model_without_ddp.get_CLA_loss(shallow_feas, deep_feas, scales=[1, 2], consis_thresh=args.consis_thresh)
                mse_cost = model_without_ddp.get_CIA_loss(all_feas, bz, layer_idx=3)
            else:
                cls_dis_cost, sep_dis_cost = model_without_ddp.get_PA_loss(proto_acts)

            # Evaluation statistics
            _, predicted = torch.max(logits.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1

        # Compute gradient and do SGD step
        if is_train:
            if epoch < args.warmup_epochs:   # Freeze the backbone
                loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['orth'] * ortho_cost)
            elif epoch >= args.warmup_epochs and epoch < args.proto_epochs:  # Unfreeze the backbone
                loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['orth'] * ortho_cost
                    + coefs['consis'] * consis_cost
                    + coefs['mse'] * mse_cost)
            elif epoch >= args.proto_epochs:    # Only train the predictor
                loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['attri'] * attributes_cost
                    + coefs['cls_dis'] * cls_dis_cost
                    + coefs['sep_dis'] * sep_dis_cost)
                    
            loss_value = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize basis vectors
            model_without_ddp.prototype_vectors.data = F.normalize(model_without_ddp.prototype_vectors, p=2, dim=1).data

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            tb_writer.add_scalars(
                main_tag="train/loss",
                tag_scalar_dict={
                    "cls": loss.item(),
                },
                global_step=iteration+it
            )
            it += 1

        # Del input
        del target
        del output
        del predicted
    
    results_loss = {'accu' : n_correct/n_examples}
    return n_correct / n_examples, results_loss


def train(model, epoch, dataloader, optimizer, tb_writer, iteration, coefs=None, args=None, log=print):
    assert(optimizer is not None)

    model.train()
    return _train_or_test(model=model, epoch=epoch, dataloader=dataloader, optimizer=optimizer, tb_writer=tb_writer,
                          iteration=iteration, coefs=coefs, args=args, log=log)


def test(model, epoch, dataloader, tb_writer, iteration, args=None, log=print):
    model.eval()
    return _train_or_test(model=model, epoch=epoch, dataloader=dataloader, optimizer=None, tb_writer=tb_writer,
                          iteration=iteration, args=args, log=log)


def warm_only_new(model):
    if hasattr(model, 'module'):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.attributes_predictor.parameters():
        p.requires_grad = False
    for p in model.class_predictor.parameters():
        p.requires_grad = False


def joint_new(model):
    if hasattr(model, 'module'):
        model = model.module
    for p in model.features.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.attributes_predictor.parameters():
        p.requires_grad = False
    for p in model.class_predictor.parameters():
        p.requires_grad = False


def final_new(model):
    if hasattr(model, 'module'):
        model = model.module
    # for p in model.features.parameters():
    #     p.requires_grad = False
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    for p in model.attributes_predictor.parameters():
        p.requires_grad = True
    for p in model.class_predictor.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True