import numpy as np
import torch
import argparse

class Conv3D(object):
    """
    The input tensor: N*Cin*D*H*W np.ndarray
    The weight: Cout*Cin*d*h*w np.ndarray
    Other inputs: stride: [stride_h: int, stride_w: int]
    padding: zero padding size, [padding_h: int, padding_w: int]
    dilation: [dilation_h: int, dilation_w: int], dilation rate
    Bias is not considered here
    """
    def __init__(self, weight, stride, padding, do_dilation, dilation):
        self.weight = weight
        self.out_channel, self.in_channel, self.kernel_d, self.kernel_h, self.kernel_w = self.weight.shape
        self.stride = stride if isinstance(stride, list) else [stride, stride, stride]
        self.padding = padding if isinstance(padding, list) else [padding, padding, padding]
        if do_dilation:
            self.dilation = dilation if isinstance(dilation, list) else [dilation, dilation, dilation]
            self.dilated_kernel_d = (self.kernel_d - 1) * self.dilation[0] + 1
            self.dilated_kernel_h = (self.kernel_h - 1) * self.dilation[1] + 1
            self.dilated_kernel_w = (self.kernel_w - 1) * self.dilation[2] + 1
            self.dilated_kernel = np.zeros([self.out_channel, self.in_channel, self.dilated_kernel_d, self.dilated_kernel_h, self.dilated_kernel_w])
            for d_iter in range(self.kernel_d):
                for h_iter in range(self.kernel_h):
                    for w_iter in range(self.kernel_w):
                        self.dilated_kernel[:, :, 
                                            d_iter*self.dilation[0], 
                                            h_iter*self.dilation[1], 
                                            w_iter*self.dilation[2]] = self.weight[:, :, d_iter, h_iter, w_iter]
        else:
            self.dilated_kernel = self.weight
        self.flattened_dilated_kernel = np.zeros([self.in_channel * self.dilated_kernel_d * self.dilated_kernel_h * self.dilated_kernel_w,
                                                  self.out_channel])
        for cout_iter in range(self.out_channel):
            for d_iter in range(self.dilated_kernel_d):
                for h_iter in range(self.dilated_kernel_h):
                    for w_iter in range(self.dilated_kernel_w):
                        self.flattened_dilated_kernel[self.in_channel*(d_iter*self.dilated_kernel_h*self.dilated_kernel_w + h_iter*self.dilated_kernel_w + w_iter):self.in_channel*(d_iter*self.dilated_kernel_h*self.dilated_kernel_w + h_iter*self.dilated_kernel_w + w_iter + 1), 
                                                      cout_iter] = self.dilated_kernel[cout_iter, :, d_iter, h_iter, w_iter]

    def __call__(self, input_tensor, method):
        _, in_channel, _, _, _ = input_tensor.shape
        assert in_channel == self.in_channel, "[ERROR] input channel not matched."
        if method == "naive":
            return self.infer_naive(input_tensor)
        if method == "img2col":
            return self.infer_img2col(input_tensor)
        if method == "winograde":
            return self.infer_winograde(input_tensor)
        raise NotImplementedError
    
    def infer_naive(self, input_tensor):
        batch_size, in_channel, input_D, input_H, input_W = input_tensor.shape
        padded_input_D = input_D + 2 * self.padding[0]
        padded_input_H = input_H + 2 * self.padding[1]
        padded_input_W = input_W + 2 * self.padding[2]
        padded_input = np.zeros([batch_size, in_channel, padded_input_D, padded_input_H, padded_input_W])
        padded_input[:, :, 
                     self.padding[0]:self.padding[0]+input_D,
                     self.padding[1]:self.padding[1]+input_H,
                     self.padding[2]:self.padding[2]+input_W] = input_tensor
        output_tensor = np.zeros([batch_size, self.out_channel,
                                  (padded_input_D - self.dilated_kernel_d) // self.stride[0] + 1,
                                  (padded_input_H - self.dilated_kernel_h) // self.stride[1] + 1,
                                  (padded_input_W - self.dilated_kernel_w) // self.stride[2] + 1])
        _dilated_kernel = np.repeat(self.dilated_kernel[np.newaxis, :, :, :, :, :], batch_size, axis=0)
        for d_iter in range(output_tensor.shape[2]):
            for h_iter in range(output_tensor.shape[3]):
                for w_iter in range(output_tensor.shape[4]):
                    # calculate output_tensor[:, :, d_iter, h_iter, w_iter]
                    respective_field = padded_input[:, :,
                                                    d_iter * self.stride[0]:d_iter * self.stride[0] + self.dilated_kernel_d,
                                                    h_iter * self.stride[1]:h_iter * self.stride[1] + self.dilated_kernel_h,
                                                    w_iter * self.stride[2]:w_iter * self.stride[2] + self.dilated_kernel_w]
                    respective_field = np.repeat(respective_field[:, np.newaxis, :, :, :, :], self.out_channel, axis=1)
                    output_tensor[:, :, d_iter, h_iter, w_iter] = np.sum(_dilated_kernel * respective_field, axis=(-1, -2, -3, -4))
        return output_tensor

    def infer_img2col(self, input_tensor):
        batch_size, in_channel, input_D, input_H, input_W = input_tensor.shape
        padded_input_D = input_D + 2 * self.padding[0]
        padded_input_H = input_H + 2 * self.padding[1]
        padded_input_W = input_W + 2 * self.padding[2]
        padded_input = np.zeros([batch_size, in_channel, padded_input_D, padded_input_H, padded_input_W])
        padded_input[:, :, 
                     self.padding[0]:self.padding[0]+input_D,
                     self.padding[1]:self.padding[1]+input_H,
                     self.padding[2]:self.padding[2]+input_W] = input_tensor
        output_tensor_size = [batch_size, self.out_channel,
                              (padded_input_D - self.dilated_kernel_d) // self.stride[0] + 1,
                              (padded_input_H - self.dilated_kernel_h) // self.stride[1] + 1,
                              (padded_input_W - self.dilated_kernel_w) // self.stride[2] + 1]
        flattened_input_tensor = np.zeros([batch_size*output_tensor_size[2]*output_tensor_size[3]*output_tensor_size[4],
                                           self.flattened_dilated_kernel.shape[0]])
        for batch_iter in range(batch_size):
            for d_iter in range(output_tensor_size[2]):
                for h_iter in range(output_tensor_size[3]):
                    for w_iter in range(output_tensor_size[4]):
                        _row = batch_iter * (output_tensor_size[2]*output_tensor_size[3]*output_tensor_size[4]) + \
                            d_iter * (output_tensor_size[3]*output_tensor_size[4]) + h_iter * (output_tensor_size[4]) + w_iter
                        respective_field = padded_input[batch_iter, :,
                                                        d_iter * self.stride[0]:d_iter * self.stride[0] + self.dilated_kernel_d,
                                                        h_iter * self.stride[1]:h_iter * self.stride[1] + self.dilated_kernel_h,
                                                        w_iter * self.stride[2]:w_iter * self.stride[2] + self.dilated_kernel_w]
                        for kd_iter in range(self.dilated_kernel_d):
                            for kh_iter in range(self.dilated_kernel_h):
                                for kw_iter in range(self.dilated_kernel_w):
                                    _col = kd_iter * (self.dilated_kernel_h*self.dilated_kernel_w) + kh_iter * self.dilated_kernel_w + kw_iter
                                    flattened_input_tensor[_row, _col*self.in_channel:(_col+1)*self.in_channel] = respective_field[:, kd_iter, kh_iter, kw_iter]
        flattened_output_tensor = np.matmul(flattened_input_tensor, self.flattened_dilated_kernel)
        output_tensor = np.zeros(output_tensor_size)
        for batch_iter in range(batch_size):
            for d_iter in range(output_tensor_size[2]):
                for h_iter in range(output_tensor_size[3]):
                    for w_iter in range(output_tensor_size[4]):
                        _row = batch_iter * (output_tensor_size[2]*output_tensor_size[3]*output_tensor_size[4]) + \
                            d_iter * (output_tensor_size[3]*output_tensor_size[4]) + h_iter * (output_tensor_size[4]) + w_iter
                        output_tensor[batch_iter, :, d_iter, h_iter, w_iter] = flattened_output_tensor[_row, :]
        return output_tensor

    def infer_winograde(self, input_tensor):
        batch_size, in_channel, input_D, input_H, input_W = input_tensor.shape
        padded_input_D = input_D + 2 * self.padding[0]
        padded_input_H = input_H + 2 * self.padding[1]
        padded_input_W = input_W + 2 * self.padding[2]
        padded_input = np.zeros([batch_size, in_channel, padded_input_D, padded_input_H, padded_input_W])
        padded_input[:, :, 
                     self.padding[0]:self.padding[0]+input_D,
                     self.padding[1]:self.padding[1]+input_H,
                     self.padding[2]:self.padding[2]+input_W] = input_tensor
        output_tensor_size = [batch_size, self.out_channel,
                              (padded_input_D - self.dilated_kernel_d) // self.stride[0] + 1,
                              (padded_input_H - self.dilated_kernel_h) // self.stride[1] + 1,
                              (padded_input_W - self.dilated_kernel_w) // self.stride[2] + 1]
        flattened_input_tensor = np.zeros([batch_size*output_tensor_size[2]*output_tensor_size[3]*output_tensor_size[4],
                                           self.flattened_dilated_kernel.shape[0]])
        for batch_iter in range(batch_size):
            for d_iter in range(output_tensor_size[2]):
                for h_iter in range(output_tensor_size[3]):
                    for w_iter in range(output_tensor_size[4]):
                        _row = batch_iter * (output_tensor_size[2]*output_tensor_size[3]*output_tensor_size[4]) + \
                            d_iter * (output_tensor_size[3]*output_tensor_size[4]) + h_iter * (output_tensor_size[4]) + w_iter
                        respective_field = padded_input[batch_iter, :,
                                                        d_iter * self.stride[0]:d_iter * self.stride[0] + self.dilated_kernel_d,
                                                        h_iter * self.stride[1]:h_iter * self.stride[1] + self.dilated_kernel_h,
                                                        w_iter * self.stride[2]:w_iter * self.stride[2] + self.dilated_kernel_w]
                        for kd_iter in range(self.dilated_kernel_d):
                            for kh_iter in range(self.dilated_kernel_h):
                                for kw_iter in range(self.dilated_kernel_w):
                                    _col = kd_iter * (self.dilated_kernel_h*self.dilated_kernel_w) + kh_iter * self.dilated_kernel_w + kw_iter
                                    flattened_input_tensor[_row, _col*self.in_channel:(_col+1)*self.in_channel] = respective_field[:, kd_iter, kh_iter, kw_iter]
        # use winograde for:
        # flattened_output_tensor = np.matmul(flattened_input_tensor, self.flattened_dilated_kernel)
        raise NotImplementedError
        output_tensor = np.zeros(output_tensor_size)
        for batch_iter in range(batch_size):
            for d_iter in range(output_tensor_size[2]):
                for h_iter in range(output_tensor_size[3]):
                    for w_iter in range(output_tensor_size[4]):
                        _row = batch_iter * (output_tensor_size[2]*output_tensor_size[3]*output_tensor_size[4]) + \
                            d_iter * (output_tensor_size[3]*output_tensor_size[4]) + h_iter * (output_tensor_size[4]) + w_iter
                        output_tensor[batch_iter, :, d_iter, h_iter, w_iter] = flattened_output_tensor[_row, :]
        return output_tensor
        

def argument_parser():
    parser = argparse.ArgumentParser(description="Conv arg parser.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--input_channel", type=int, default=32)
    parser.add_argument("--output_channel", type=int, default=16)
    parser.add_argument("--input_D", type=int, default=32)
    parser.add_argument("--input_H", type=int, default=16)
    parser.add_argument("--input_W", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=[2, 1, 2])
    parser.add_argument("--dilation", type=int, default=[1, 2, 2])
    parser.add_argument("--padding", type=int, default=[4, 4, 4])
    parser.add_argument("--test_round", type=int, default=5)
    return parser.parse_args()

if __name__=='__main__':
    args = argument_parser()
    torch_conv3d = torch.nn.Conv3d(args.input_channel, args.output_channel, args.kernel_size, 
                                   stride=args.stride, dilation=args.dilation, padding=args.padding, bias=False)
    my_conv3d = Conv3D(weight=torch_conv3d.weight.detach().numpy(), 
                       stride=args.stride, padding=args.padding, 
                       do_dilation=True, dilation=args.dilation)
    for _ in range(args.test_round):
        input_tensor = torch.randn(args.batch_size, args.input_channel, args.input_D, args.input_H, args.input_W)
        ref_output_tensor = torch_conv3d(input_tensor).detach().numpy()
        output_tensor_naive = my_conv3d(input_tensor.numpy(), method="naive")
        output_tensor_img2col = my_conv3d(input_tensor.numpy(), method="img2col")
        print(f"Naive version: maximum abs error:\t{np.max(np.abs(output_tensor_naive-ref_output_tensor))}.")
        print(f"img2col version: maximum abs error:\t{np.max(np.abs(output_tensor_img2col-ref_output_tensor))}.")
