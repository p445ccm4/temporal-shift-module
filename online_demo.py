import numpy as np
import cv2
import os
from typing import Tuple
import io
import tvm
import tvm.relay
import time
import onnx
import torch
import torchvision
import torch.onnx
from PIL import Image, ImageOps
import tvm.contrib.graph_runtime as graph_runtime
from online_demo.mobilenet_v2_tsm import MobileNetV2

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True


def torch2tvm_module(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    torch_module.eval()
    input_names = []
    input_shapes = {}
    with torch.no_grad():
        for index, torch_input in enumerate(torch_inputs):
            name = "i" + str(index)
            input_names.append(name)
            input_shapes[name] = torch_input.shape
        buffer = io.BytesIO()
        torch.onnx.export(torch_module, torch_inputs, buffer, input_names=input_names,
                          output_names=["o" + str(i) for i in range(len(torch_inputs))], opset_version=10)
        outs = torch_module(*torch_inputs)
        buffer.seek(0, 0)
        onnx_model = onnx.load_model(buffer)
        from onnxsim import simplify
        onnx_model, success = simplify(onnx_model)  # this simplifier removes conversion bugs.
        assert success
        relay_module, params = tvm.relay.frontend.from_onnx(onnx_model, shape=input_shapes)
    with tvm.relay.build_config(opt_level=3):
        graph, tvm_module, params = tvm.relay.build(relay_module, target, params=params)
    return graph, tvm_module, params


def torch2executor(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    prefix = f"mobilenet_tsm_tvm_{target}"
    lib_fname = f'{prefix}.tar'
    graph_fname = f'{prefix}.json'
    params_fname = f'{prefix}.params'
    if os.path.exists(lib_fname) and os.path.exists(graph_fname) and os.path.exists(params_fname):
        with open(graph_fname, 'rt') as f:
            graph = f.read()
        tvm_module = tvm.module.load(lib_fname)
        params = tvm.relay.load_param_dict(bytearray(open(params_fname, 'rb').read()))
    else:
        graph, tvm_module, params = torch2tvm_module(torch_module, torch_inputs, target)
        tvm_module.export_library(lib_fname)
        with open(graph_fname, 'wt') as f:
            f.write(graph)
        with open(params_fname, 'wb') as f:
            f.write(tvm.relay.save_param_dict(params))

    ctx = tvm.gpu() if target.startswith('cuda') else tvm.cpu()
    graph_module = graph_runtime.create(graph, tvm_module, ctx)
    for pname, pvalue in params.items():
        graph_module.set_input(pname, pvalue)

    def executor(inputs: Tuple[tvm.nd.NDArray]):
        for index, value in enumerate(inputs):
            graph_module.set_input(index, value)
        graph_module.run()
        return tuple(graph_module.get_output(index) for index in range(len(inputs)))

    return executor, ctx


def get_executor(use_gpu=True):
    torch_module = MobileNetV2(n_class=2)
    model_dict = torch_module.state_dict()
    replace_dict = []
    sd = torch.load("checkpoint/TSM_bb-dataset-cropped_RGB_mobilenetv2_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar")['state_dict']
    for k, v in sd.items():
        if k not in model_dict and k.replace('module.base_model.features', 'features').replace('.net', '') in model_dict:
            print('=> Load after removing module.base_model: ', k)
            replace_dict.append((k, k.replace('module.base_model.features', 'features').replace('.net', '')))
        if k not in model_dict and k.replace('module.new_fc', 'classifier') in model_dict:
            print('=> Load after removing module.base_model: ', k)
            replace_dict.append((k, k.replace('module.new_fc', 'classifier')))
    for k, k_new in replace_dict:
        sd[k_new] = sd.pop(k)
    keys1 = set(list(sd.keys()))
    keys2 = set(list(model_dict.keys()))
    set_diff = (keys1 - keys2) | (keys2 - keys1)
    print('#### Notice: keys that failed to load: {}'.format(set_diff))
    model_dict.update(sd)
    torch_module.load_state_dict(model_dict)

    torch_inputs = (torch.rand(1, 3, 224, 224),
                    torch.zeros([1, 3, 56, 56]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 4, 28, 28]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 8, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 12, 14, 14]),
                    torch.zeros([1, 20, 7, 7]),
                    torch.zeros([1, 20, 7, 7]))
    if use_gpu:
        target = 'cuda'
    else:
        target = 'llvm -mcpu=cortex-a72 -target=armv7l-linux-gnueabihf'
    return torch2executor(torch_module, torch_inputs, target)

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupCenterPad(object):
    '''
    Pad a batch of images with different sizes into size (desired_size).
    Usage:
        gcp = GroupCenterPad(224)
        imgs = gcp(imgs)
    '''
    def __init__(self, desired_size=224):
        self.desired_size = (desired_size, desired_size) if type(desired_size) is int else desired_size

    def __call__(self, img_group):
        ret = []
        for img in img_group:
            w, h = img.size
            w_pad = max((h - w) // 2, 0)
            h_pad = max((w - h) // 2, 0)
            # img = torchvision.transforms.Pad((w_pad, h_pad))(img)
            img = torchvision.transforms.Resize(self.desired_size)(img)
            ret.append(img)
        return ret


def get_transform():
    transform = torchvision.transforms.Compose([
        GroupCenterPad(224),
        Stack(False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform

catigories = [
    "Abnormal", "Normal"
]


n_still_frame = 0


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer

    # mask out illegal action
    if idx_ in [7, 8, 21, 22, 3]:
        idx_ = history[-1]

    # # use only single no action class
    # if idx_ == 0:
    #     idx_ = 2

    # history smoothing
    if idx_ != history[-1]:
        if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
            idx_ = history[-1]

    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history


WINDOW_NAME = 'Video Gesture Recognition'


def main():
    print("Open camera...")
    # cap = cv2.VideoCapture("online_demo/test_video/img_%03d.jpg", )

    # # set a lower resolution for speed up
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    video_dir = "online_demo/test_video/"
    frame_paths = [os.path.join(video_dir, file) for file in os.listdir(video_dir)]

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    print("Build transformer...")
    transform = get_transform()
    print("Build Executor...")
    executor, ctx = get_executor()
    buffer = (
        tvm.nd.empty((1, 3, 56, 56), ctx=ctx),
        tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
        tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
        tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
        tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
    )
    idx = 0
    history = [0, 0]
    history_logit = []
    history_timing = []

    i_frame = -1

    print("Ready!")
    # while True:
    for frame_path in frame_paths:
        i_frame += 1
        # _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        img = cv2.imread(frame_path)
        if i_frame % 1 == 0:  # skip every other frame to obtain a suitable frame rate
            t1 = time.time()
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
            img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=ctx)
            inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + buffer
            outputs = executor(inputs)
            feat, buffer = outputs[0], outputs[1:]
            assert isinstance(feat, tvm.nd.NDArray)

            if SOFTMAX_THRES > 0:
                feat_np = feat.asnumpy().reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
                if max(softmax) > SOFTMAX_THRES:
                    idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

            if HISTORY_LOGIT:
                history_logit.append(feat.asnumpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = process_output(idx_, history)

            t2 = time.time()
            print(f"{index} {catigories[idx]}")


            current_time = t2 - t1

        img = cv2.resize(img, (480, 1280))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)


        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    # cap.release()
    cv2.destroyAllWindows()


main()
