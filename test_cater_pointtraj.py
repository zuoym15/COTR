import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from COTR.models import build_model
from COTR.utils import debug_utils, utils
from COTR.datasets import cater_pointtraj_dataset_new
from COTR.trainers.cotr_trainer import COTRTrainer
from COTR.global_configs import general_config
from COTR.options.options import *
from COTR.options.options_utils import *

from COTR.datasets.utils import improc, misc

from tensorboardX import SummaryWriter

EPS = 1e-6
def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so
    # returns shape-1
    # axis can be a list of axes
    for (a,b) in zip(x.size(), mask.size()):
        if not b==1:
            assert(a==b) # some shape mismatch!
    # assert(x.size() == mask.size())
    prod = x*mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS+torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS+torch.sum(mask, dim=dim, keepdim=keepdim)
        
    mean = numer/denom
    return mean

def sequence_loss(flow_preds, flow_gt, vis, valid, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    B, S, N, D = flow_gt.shape
    assert(D==2)
    B, S, N = vis.shape
    B, N = valid.shape
    
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs() # B, S, N, 2
        i_loss = torch.mean(i_loss, dim=[1,3]) # B, N
        flow_loss += i_weight * reduce_masked_mean(i_loss, valid)

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=-1).sqrt() # B, S, N

    epe_vis = reduce_masked_mean(epe, valid.unsqueeze(1)*vis[:,1:])
    epe_inv = reduce_masked_mean(epe, valid.unsqueeze(1)*(1.0-vis[:,1:]))

    epe_inv2inv = reduce_masked_mean(epe, valid.unsqueeze(1) * (1.0 - (vis[:,1:] + vis[:,:-1]).clamp(0,1)))
    
    metrics = {
        'epe': epe.mean().item(),
        'epe_vis': epe_vis.item(),
        'epe_inv': epe_inv.item(),
        'epe_inv2inv': epe_inv2inv.item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        '10px': (epe < 10).float().mean().item(),
        '30px': (epe < 30).float().mean().item(),
    }

    return flow_loss, metrics

def run(opt):
    # model
    model = build_model(opt)
    model = model.cuda()
    weights = torch.load(opt.load_weights_path, map_location='cuda')['model_state_dict']
    utils.safe_load_weights(model, weights)
    model = model.eval()

    # data
    val_dset = cater_pointtraj_dataset_new.CATERPointTrajDataset(opt, dset='v', use_augs=False)
    val_dataloader = DataLoader(val_dset, batch_size=1,
                                shuffle=False, num_workers=0)
    val_iterloader = iter(val_dataloader)

    # writer
    log_dir = 'logs_test_cater_pointtraj'
    model_name = 't00'
    model_name = 't01'
    model_name = 't02' # s=8
    writer = SummaryWriter(log_dir + '/' + model_name, max_queue=10, flush_secs=60)
    log_freq = 50 #200

    epe_pool = misc.SimplePool(1000, version='np')
    epe_vis_pool = misc.SimplePool(1000, version='np')
    epe_inv_pool = misc.SimplePool(1000, version='np')
    epe_inv2inv_pool = misc.SimplePool(1000, version='np')

    global_step = 300000
    max_iter = 350000
    while global_step < max_iter:
        global_step += 1
        print("{0} / {1}".format(global_step, max_iter))
        try:
            sample = next(val_iterloader)
        except StopIteration:
            val_iterloader = iter(val_dataloader)
            sample = next(val_iterloader)

        if global_step % 50 != 0:
            continue

        sw = improc.Summ_writer(
            writer=writer,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        with torch.no_grad():
            rgb = sample['rgbs'][0].float().cuda() # S, C, H, W
            keypoints_xy = sample['all_xy'][0].cuda() # S, N, 2
            visibles = sample['all_vis'][0].cuda() # S, N

            rgb = rgb / 255.0
            mean_ = torch.as_tensor([0.485, 0.456, 0.406]).float().reshape(1,3,1,1).to(rgb.device)
            std_ = torch.as_tensor([0.229, 0.224, 0.225]).float().reshape(1,3,1,1).to(rgb.device)
            rgb = (rgb - mean_) / std_

            #rgb = rgb[:2]
            #keypoints_xy = keypoints_xy[:2]
            #visibles = visibles[:2]

            rgb_orig = rgb.clone()
            S, C, H, W = rgb_orig.shape
            img = F.interpolate(rgb_orig, (256, 256))

            kp_xy0 = keypoints_xy[0].clone() # N, 2
            kp_xy0[:, 0] *= (0.5 / float(W))
            kp_xy0[:, 1] *= (1.0 / float(H))
            query = kp_xy0.unsqueeze(0)

            preds = []
            for i in range(S-1):
            #for i in range(1):
                img0 = img[0].clone() # 3, H, W
                img1 = img[i+1].clone() # 3, H, W

                sbs_img = torch.cat([img0, img1], dim=-1) # 3, H, 2*W

                pred = model(sbs_img.unsqueeze(0), query)['pred_corrs']
                pred[:,:,0] -= 0.5
                #query = pred.clone()

                pred[:,:,0] *= float(2*W)
                pred[:,:,1] *= float(H)
                flow = pred - keypoints_xy[0:1]
                preds.append(flow.clone())

            preds = [torch.cat(preds, 0).unsqueeze(0)]
            targets = keypoints_xy[1:] - keypoints_xy[0:1].unsqueeze(0)
            vis = visibles.unsqueeze(0)
            valid = torch.ones_like(vis[:,0])

            loss, metrics = sequence_loss(preds, targets, vis, valid)
            # loss1, metrics1 = sequence_loss([preds[0][:,:2]], targets[:,:2], vis[:,:3], valid)

            trajs = keypoints_xy.unsqueeze(0)
            
            trajs_e = torch.cat([trajs[:,0:1], trajs[:,0:1] + preds[-1]], dim=1)
            #trajs_e = torch.cat([pred_orig.unsqueeze(0), pred_orig + preds[-1]], dim=1)

            if sw is not None:
                mean_ = torch.as_tensor([0.485, 0.456, 0.406]).float().reshape(1,1,3,1,1).to(img.device)
                std_ = torch.as_tensor([0.229, 0.224, 0.225]).float().reshape(1,1,3,1,1).to(img.device)
                rgbs = sample['rgbs'].float().cuda()
                #rgbs = (rgbs * std_) + mean_
                #rgbs = rgbs.clamp(0,1) - 0.5

                rgbs = rgbs / 255.0 - 0.5

                #sw.summ_gif('inputs/rgbs', rgbs)

                sw.summ_traj2ds_on_rgbs('inputs/trajs_on_rgbs', trajs, rgbs)
                sw.summ_traj2ds_on_rgbs('inputs/trajs_on_black', trajs, torch.ones_like(rgbs)*-0.5)                
                sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e, rgbs)
                sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e, torch.ones_like(rgbs)*-0.5)

                epe_pool.update([metrics['epe']])
                sw.summ_scalar('pooled/epe', epe_pool.mean())
                epe_vis_pool.update([metrics['epe_vis']])
                sw.summ_scalar('pooled/epe_vis', epe_vis_pool.mean())
                epe_inv_pool.update([metrics['epe_inv']])
                sw.summ_scalar('pooled/epe_inv', epe_inv_pool.mean())
                if metrics['epe_inv2inv'] > 0:
                    epe_inv2inv_pool.update([metrics['epe_inv2inv']])
                    sw.summ_scalar('pooled/epe_inv2inv', epe_inv2inv_pool.mean())
                print("epe={0}, vis={1}, inv={2}, inv={3}", epe_pool.mean(), epe_vis_pool.mean(), epe_inv_pool.mean(), epe_inv2inv_pool.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
    parser.add_argument('--num_kp', type=int, default=256)
    parser.add_argument('--bidirectional', type=str2bool, default=False, help='left2right and right2left')
    parser.add_argument('--need_rotation', type=str2bool, default=False, help='rotation augmentation')

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    layer_2_channels = {'layer1': 256,
                        'layer2': 512,
                        'layer3': 1024,
                        'layer4': 2048, }
    opt.dim_feedforward = layer_2_channels[opt.layer]
    if opt.load_weights:
        opt.load_weights_path = os.path.join(opt.out_dir, opt.load_weights, 'checkpoint.pth.tar')
    print_opt(opt)
    run(opt)
              
