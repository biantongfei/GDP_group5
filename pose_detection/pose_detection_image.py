import platform
import torch
import os
import argparse
from tqdm import tqdm
import numpy as np
import time

from Alphapose.alphapose.utils.config import update_config
from Alphapose.alphapose.utils.detector import DetectionLoader
from Alphapose.detector.apis import get_detector
from Alphapose.alphapose.models import builder
from Alphapose.alphapose.utils.writer import DataWriter
from Alphapose.alphapose.utils.vis import getTime

print('finish import')


def print_finish_info(args):
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print(
            '===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def pose_detection_image(img, format='coco'):
    # check input params
    parser = argparse.ArgumentParser(description='AlphaPose Demo')
    parser.add_argument('--cfg', type=str,
                        default='Alphapose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
                        help='experiment configure file name')
    parser.add_argument('--checkpoint', type=str,
                        default='Alphapose/pretrained_models/fast_res50_256x192.pth',
                        help='checkpoint file name')
    parser.add_argument('--sp', default=False, action='store_true',
                        help='Use single process for pytorch')
    parser.add_argument('--detector', dest='detector',
                        help='detector name', default="yolo")
    parser.add_argument('--detfile', dest='detfile',
                        help='detection result file', default="")
    parser.add_argument('--indir', dest='inputpath',
                        help='image-directory', default="")
    parser.add_argument('--list', dest='inputlist',
                        help='image-list', default="")
    parser.add_argument('--image', dest='inputimg',
                        help='image-name', default=img)
    parser.add_argument('--outdir', dest='outputpath',
                        help='output-directory', default="output/images")
    parser.add_argument('--save_img', default=True, action='store_true',
                        help='save result as image')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='visualize image')
    parser.add_argument('--showbox', default=False, action='store_true',
                        help='visualize human bbox')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='add speed profiling at screen output')
    parser.add_argument('--format', type=str, default=format,
                        help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
    parser.add_argument('--min_box_area', type=int, default=0,
                        help='min box area to filter out')
    parser.add_argument('--detbatch', type=int, default=5,
                        help='detection batch size PER GPU')
    parser.add_argument('--posebatch', type=int, default=64,
                        help='pose estimation maximum batch size PER GPU')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                        help='save the result json as coco format, using image index(int) instead of image name(str)')
    parser.add_argument('--gpus', type=str, dest='gpus', default="2",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--qsize', type=int, dest='qsize', default=1,
                        help='the length of result buffer, where reducing it will lower requirement of cpu memory')
    parser.add_argument('--flip', default=False, action='store_true',
                        help='enable flip testing')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print detail information')
    parser.add_argument('--video', dest='video',
                        help='video-name', default="")
    parser.add_argument('--webcam', dest='webcam', type=int,
                        help='webcam number', default=-1)
    parser.add_argument('--save_video', dest='save_video',
                        help='whether to save rendered video', default=False, action='store_true')
    parser.add_argument('--vis_fast', dest='vis_fast',
                        help='use fast rendering', action='store_true', default=False)

    args = parser.parse_args()
    cfg = update_config(args.cfg)

    if platform.system() == 'Windows':
        args.sp = True

    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    args.detbatch = args.detbatch * len(args.gpus)
    args.posebatch = args.posebatch * len(args.gpus)
    args.tracking = False
    args.pose_flow = False
    args.pose_track = False

    # if not args.sp:
    #     torch.multiprocessing.set_start_method('forkserver', force=True)
    #     torch.multiprocessing.set_sharing_strategy('file_system')

    inputimg = args.inputimg
    args.inputpath = os.path.split(inputimg)[0]
    im_names = [os.path.split(inputimg)[1]]

    # Load detection loader
    det_loader = DetectionLoader(im_names, get_detector(args), cfg, args, batchSize=args.detbatch,
                                 queueSize=args.qsize)
    det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.checkpoint,))
    print(args.device, 'device--------')
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = args.qsize
    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()
    data_len = det_loader.length
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)

                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']),
                        pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info(args)
        while (writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
        writer.stop()
        det_loader.stop()
    except Exception as e:
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        pass
    except KeyboardInterrupt:
        print_finish_info(args)
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while (writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(
                    writer.count()) + ' images in the queue...')
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()


if __name__ == '__main__':
    pose_detection_image('input/images/train_1003.jpg')
