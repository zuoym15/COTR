import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from COTR.models import build_model
from COTR.utils import debug_utils, utils
from COTR.datasets import cater_pointtraj_dataset
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
    val_dset = cater_pointtraj_dataset.CATERPointTrajDataset(opt, 'val', use_augs=False)
    val_dataloader = DataLoader(val_dset, batch_size=1,
                                shuffle=False, num_workers=0)
    val_iterloader = iter(val_dataloader)

    # writer
    log_dir = 'logs_test_cotr_iterative'
    model_name = 't01'
    writer = SummaryWriter(log_dir + '/' + model_name, max_queue=10, flush_secs=60)
    log_freq = 1 #200

    epe_pool = misc.SimplePool(1000, version='np')
    epe_vis_pool = misc.SimplePool(1000, version='np')
    epe_inv_pool = misc.SimplePool(1000, version='np')
    epe_inv2inv_pool = misc.SimplePool(1000, version='np')

    import glob
    imgs_png = glob.glob("/home/zhaoyuaf/research/COTR/sample_data/imgs/*.png")
    imgs_jpg = glob.glob("/home/zhaoyuaf/research/COTR/sample_data/imgs/*.jpg")
    imgs = imgs_png + imgs_jpg

    global_step = 17 
    max_iter = len(imgs)
    while global_step < max_iter:
        global_step += 1
        print("{0} / {1}".format(global_step, max_iter))
        try:
            sample = next(val_iterloader)
        except StopIteration:
            val_iterloader = iter(val_dataloader)
            sample = next(val_iterloader)

        sw = improc.Summ_writer(
            writer=writer,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq/2),
            just_gif=True)

        from PIL import Image
        rgb = np.array(Image.open(imgs[global_step-1]).convert('RGB')).astype(float) / 255
        rgb = F.interpolate(torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float(), (384,384)).cuda()

        offset_x = 10
        offset_y = 0

        mean_ = torch.as_tensor([0.485, 0.456, 0.406]).float().reshape(1,3,1,1).to(rgb.device)
        std_ = torch.as_tensor([0.229, 0.224, 0.225]).float().reshape(1,3,1,1).to(rgb.device)

        rgb = (rgb - mean_) / std_
        sbs_img = torch.cat([rgb, rgb], dim=-1)

        query_x = torch.rand(256) * 0.5
        query_y = torch.rand(256)
        query = torch.stack([query_x, query_y], dim=1).unsqueeze(0).cuda()
        query_pix = query.clone()
        query_pix[:,:,0] *= 512
        query_pix[:,:,1] *= 256

        trajs_e = []
        trajs_e.append(query_pix)

        rgbs = []

        for i in range(9):
            rgb0 = rgb[:,:,i*offset_y:i*offset_y+256, i*offset_x:i*offset_x+256]
            rgb1 = rgb[:,:,(i+1)*offset_y:(i+1)*offset_y+256, (i+1)*offset_x:(i+1)*offset_x+256]
            if i == 0:
                rgbs.append(rgb0)
                rgbs.append(rgb1)
            else:
                rgbs.append(rgb1)
            sbs_img = torch.cat([rgb0, rgb1], dim=-1)
            with torch.no_grad():
                pred = model(sbs_img, query)['pred_corrs']

            pred[:,:,0] -= 0.5

            query = pred.clone()

            pred[:,:,0] *= 512
            pred[:,:,1] *= 256
            
            trajs_e.append(pred.clone())
            import ipdb; ipdb.set_trace()
        trajs_e = torch.cat(trajs_e, dim=0).unsqueeze(0)

        if sw is not None:
            #rgbs = rgb.repeat(10, 1, 1, 1).unsqueeze(0)
            rgbs = torch.cat(rgbs, dim=0).unsqueeze(0)
            mean_ = torch.as_tensor([0.485, 0.456, 0.406]).float().reshape(1,1,3,1,1).to(rgbs.device)
            std_ = torch.as_tensor([0.229, 0.224, 0.225]).float().reshape(1,1,3,1,1).to(rgbs.device)

            rgbs = (rgbs * std_) + mean_
            rgbs = rgbs.clamp(0,1) - 0.5

            #sw.summ_gif('inputs/rgbs', rgbs)

            sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e, rgbs)
            sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e, torch.ones_like(rgbs)*-0.5)

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
              
