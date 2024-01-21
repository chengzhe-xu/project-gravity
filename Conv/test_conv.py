import numpy as np
import torch
import argparse

class Conv2D(object):
    """
    The input tensor: N*Cin*H*W np.ndarray
    The weight: Cout*Cin*h*w np.ndarray
    Other inputs: stride: [stride_h: int, stride_w: int]
    padding: zero padding size, [padding_h: int, padding_w: int]
    dilation: [dilation_h: int, dilation_w: int], dilation rate
    Bias is not considered here
    """
    def __init__(self, weight, stride, padding, do_dilation, dilation):
        self.weight = weight
        self.out_channel, self.in_channel, self.weight_h, self.weight_w = weight.shape
        self.stride = [stride, stride] if isinstance(stride, int) else stride
        self.padding = [padding, padding] if isinstance(padding, int) else padding
        self.dilation = [dilation, dilation] if isinstance(dilation, int) else dilation
        # dilate the weight kernel
        if do_dilation:
            dilated_kernel_size = [self.out_channel, self.in_channel,
                                (self.weight_h-1)*self.dilation[0]+1,
                                (self.weight_w-1)*self.dilation[1]+1]
            self.dilated_kernel = np.zeros(dilated_kernel_size)
            for i in range(self.weight_h):
                for j in range(self.weight_w):
                    self.dilated_kernel[:, :, i*self.dilation[0], j*self.dilation[1]] = self.weight[:, :, i, j]
        else:
            self.dilated_kernel = weight
        # flatten dilated kernel
        self.dilated_kernel_img2col = np.zeros([self.dilated_kernel.shape[2]*self.dilated_kernel.shape[3]*self.in_channel, 
                                                self.out_channel])
        for out_channel_iter in range(self.out_channel):
            for h_iter in range(self.dilated_kernel.shape[2]):
                for w_iter in range(self.dilated_kernel.shape[3]):
                    row_num = (h_iter*self.dilated_kernel.shape[3]+w_iter)*self.in_channel
                    self.dilated_kernel_img2col[row_num:row_num+self.in_channel, 
                                                out_channel_iter] = self.dilated_kernel[out_channel_iter, :, h_iter, w_iter]
    
    def __call__(self, input_tensor, method):
        _, in_channel, _, _ = input_tensor.shape
        assert in_channel == self.in_channel, "[ERROR] input channel not matched."
        if method == "naive":
            return self.infer_naive(input_tensor)
        if method == "img2col":
            return self.infer_img2col(input_tensor)
        if method == "winograde":
            return self.infer_winograde(input_tensor)
        raise NotImplementedError
    
    def infer_naive(self, input_tensor):
        batch_size, _, input_H, input_W = input_tensor.shape
        # pad the input
        padded_input = np.zeros([batch_size, self.in_channel, input_H+2*self.padding[0], input_W+2*self.padding[1]])
        padded_input[:, :, self.padding[0]:self.padding[0]+input_H, self.padding[1]:self.padding[1]+input_W] = input_tensor
        # calculate the output tensor shape
        output_tensor = np.zeros([
            batch_size, self.out_channel,
            (padded_input.shape[2] - self.dilated_kernel.shape[2]) // self.stride[0] +1,
            (padded_input.shape[3] - self.dilated_kernel.shape[3]) // self.stride[1] +1,
        ])
        kernel_ = np.repeat(self.dilated_kernel[np.newaxis, :, :, :, :], batch_size, axis=0)
        for output_h in range(output_tensor.shape[2]):
            for output_w in range(output_tensor.shape[3]):
                # calculate output_tensor[:, :, output_h, output_w]
                # calculation shape N, Cout, Cin, kernel_h, kernel_w
                # and sum over Cin, kernel_h, kernel_w
                respective_field = padded_input[:, :, output_h*self.stride[0]:output_h*self.stride[0]+self.dilated_kernel.shape[2],
                                                output_w*self.stride[1]:output_w*self.stride[1]+self.dilated_kernel.shape[3]]
                respective_field = np.repeat(respective_field[:, np.newaxis, :, :, :], self.out_channel, axis=1)
                output_tensor[:, :, output_h, output_w] = np.sum(kernel_ * respective_field, axis=(-1, -2, -3))
        return output_tensor
    
    def infer_img2col(self, input_tensor):
        batch_size, _, input_H, input_W = input_tensor.shape
        # pad the input
        padded_input = np.zeros([batch_size, self.in_channel, input_H+2*self.padding[0], input_W+2*self.padding[1]])
        padded_input[:, :, self.padding[0]:self.padding[0]+input_H, self.padding[1]:self.padding[1]+input_W] = input_tensor
        # the flattened matrix size
        output_H = (padded_input.shape[2] - self.dilated_kernel.shape[2]) // self.stride[0] +1
        output_W = (padded_input.shape[3] - self.dilated_kernel.shape[3]) // self.stride[1] +1
        output_tensor = np.zeros([batch_size, self.out_channel, output_H, output_W])
        padded_input_img2col = np.zeros([batch_size * output_H * output_W, 
                                         self.dilated_kernel.shape[2] * self.dilated_kernel.shape[3] * self.in_channel])
        # flatten padded input
        for batch_iter in range(batch_size):
            for h_iter in range(output_H):
                for w_iter in range(output_W):
                    row_num = batch_iter * output_H * output_W + h_iter * output_W + w_iter
                    for k_h_iter in range(self.dilated_kernel.shape[2]):
                        for k_w_iter in range(self.dilated_kernel.shape[3]):
                            col_num = (k_h_iter*self.dilated_kernel.shape[3]+k_w_iter)*self.in_channel
                            padded_input_img2col[row_num, col_num:col_num+self.in_channel] = padded_input[batch_iter, :, 
                                                                                                          h_iter*self.stride[0]+k_h_iter,
                                                                                                          w_iter*self.stride[1]+k_w_iter]
        output_img2col = np.matmul(padded_input_img2col, self.dilated_kernel_img2col)
        # unflatten output tensor
        for batch_iter in range(batch_size):
            for h_iter in range(output_H):
                for w_iter in range(output_W):
                    output_tensor[batch_iter, :, h_iter, w_iter] = output_img2col[batch_iter*output_H*output_W+h_iter*output_W+w_iter, :]
        return output_tensor
    
    def infer_winograde(self, input_tensor):
        batch_size, _, input_H, input_W = input_tensor.shape
        # pad the input
        padded_input = np.zeros([batch_size, self.in_channel, input_H+2*self.padding[0], input_W+2*self.padding[1]])
        padded_input[:, :, self.padding[0]:self.padding[0]+input_H, self.padding[1]:self.padding[1]+input_W] = input_tensor
        # the flattened matrix size
        output_H = (padded_input.shape[2] - self.dilated_kernel.shape[2]) // self.stride[0] +1
        output_W = (padded_input.shape[3] - self.dilated_kernel.shape[3]) // self.stride[1] +1
        output_tensor = np.zeros([batch_size, self.out_channel, output_H, output_W])
        padded_input_img2col = np.zeros([batch_size * output_H * output_W, 
                                         self.dilated_kernel.shape[2] * self.dilated_kernel.shape[3] * self.in_channel])
        # flatten padded input
        for batch_iter in range(batch_size):
            for h_iter in range(output_H):
                for w_iter in range(output_W):
                    row_num = batch_iter * output_H * output_W + h_iter * output_W + w_iter
                    for k_h_iter in range(self.dilated_kernel.shape[2]):
                        for k_w_iter in range(self.dilated_kernel.shape[3]):
                            col_num = (k_h_iter*self.dilated_kernel.shape[3]+k_w_iter)*self.in_channel
                            padded_input_img2col[row_num, col_num:col_num+self.in_channel] = padded_input[batch_iter, :, 
                                                                                                          h_iter*self.stride[0]+k_h_iter,
                                                                                                          w_iter*self.stride[1]+k_w_iter]
        # use winograde for:
        # output_img2col = np.matmul(padded_input_img2col, self.dilated_kernel_img2col)
        raise NotImplementedError

        # unflatten output tensor
        for batch_iter in range(batch_size):
            for h_iter in range(output_H):
                for w_iter in range(output_W):
                    output_tensor[batch_iter, :, h_iter, w_iter] = output_img2col[batch_iter*output_H*output_W+h_iter*output_W+w_iter, :]
        return output_tensor

def argument_parser():
    parser = argparse.ArgumentParser(description="Conv arg parser.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_channel", type=int, default=32)
    parser.add_argument("--output_channel", type=int, default=16)
    parser.add_argument("--input_H", type=int, default=128)
    parser.add_argument("--input_W", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--test_round", type=int, default=7)
    return parser.parse_args()

if __name__=='__main__':
    args = argument_parser()
    torch_conv2d = torch.nn.Conv2d(args.input_channel, args.output_channel, args.kernel_size, 
                                   stride=args.stride, dilation=args.dilation, padding=args.padding, bias=False)
    my_conv2d = Conv2D(weight=torch_conv2d.weight.detach().numpy(), 
                       stride=args.stride, padding=args.padding, 
                       do_dilation=True, dilation=args.dilation)
    for _ in range(args.test_round):
        input_tensor = torch.randn(args.batch_size, args.input_channel, args.input_H, args.input_W)
        ref_output_tensor = torch_conv2d(input_tensor).detach().numpy()
        output_tensor_naive = my_conv2d(input_tensor.numpy(), method="naive")
        output_tensor_img2col = my_conv2d(input_tensor.numpy(), method="img2col")
        print(f"Naive version: maximum abs error:\t{np.max(np.abs(output_tensor_naive-ref_output_tensor))}.")
        print(f"img2col version: maximum abs error:\t{np.max(np.abs(output_tensor_img2col-ref_output_tensor))}.")
