import argparse
import cv2
import numpy as np
from numpy.linalg import *

class GuidedInterpolationSolver(object):
    '''
    The class for the base problem: solving guided interpolation promblem,
    The Input is a channel of the target image(w*h), mask, and the guided vector field
    The Output is the corresponding channel of the output image
    '''

    def __init__(self, target, mask, mask_numpt, g_field_h, g_field_w, height, width):
        self.target = target
        self.mask = mask
        self.height = height
        self.width = width
        self.div_v = self.div(g_field_h, g_field_w)
        self.mask_numpt = (int)(mask_numpt)
    
    def div(self, v_h, v_w):
        div_ret = np.zeros_like(self.target)
        for i in range(1,self.height - 1):
            for j in range(1, self.width - 1):
                div_ret[i,j] = ( v_h[i,j] - v_h[i - 1,j] ) + ( v_w[i,j] - v_w[i, j-1] )
        return div_ret

    def solve(self):
        M = -4 * np.eye(self.mask_numpt, dtype=int)
        N = np.zeros((self.mask_numpt, 1))
        n_checked = 0
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if (self.mask[i,j]) >= 1:
                    n_checked += 1
                    omega_idx = (int)(self.mask[i,j])
                    s = self.div_v[i,j]

                    if (self.mask[i-1,j] == 0.5):
                        s = s - self.target[i-1,j]
                    elif (self.mask[i-1,j] >= 1):
                        M[(int)(self.mask[i,j]) - 1, (int)(self.mask[i-1,j]) - 1] = 1
                    
                    if (self.mask[i+1,j] == 0.5):
                        s -= self.target[i+1,j]
                    elif (self.mask[i+1,j] >= 1):
                        M[(int)(self.mask[i,j]) - 1, (int)(self.mask[i+1,j]) - 1] = 1
                    
                    if (self.mask[i,j-1] == 0.5):
                        s -= self.target[i,j-1]
                    elif (self.mask[i,j-1] >= 1):
                        M[(int)(self.mask[i,j]) - 1, (int)(self.mask[i,j-1]) - 1] = 1
                    
                    if (self.mask[i,j+1] == 0.5):
                        s -= self.target[i,j+1]
                    elif (self.mask[i,j+1] >= 1):
                        M[(int)(self.mask[i,j]) - 1, (int)(self.mask[i,j+1]) - 1] = 1
                    
                    N[(int)(self.mask[i,j]) - 1, 0] = s
        assert(n_checked == self.mask_numpt), "Not all mask points checked"
        # M = M + 1e-6 * np.eye(self.mask_numpt, dtype=int)
        # print(np.linalg.det(M))
        ans = np.linalg.solve(M, N)
        output = np.array(self.target, copy=True)
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if (self.mask[i,j]) >= 1:
                    channel_color = ans[(int)(self.mask[i,j]-1), 0]
                    if(channel_color > 255):
                        channel_color = 255
                    elif(channel_color < 0):
                        channel_color = 0
                    else:
                        channel_color = (int)(channel_color)
                    output[i,j] = channel_color
        return output

class SeamlessCloningSolver(object):
    def __init__(self, source, target, mask, m_grad):
        self.source = source
        self.target = target
        self.mask = mask
        self.m_grad = mask
        self.height, self.width, self.channels = target.shape
        print("height: " + (str)(self.height) )
        print("width: " + (str)(self.width) )
        print("mix_gradient: " + (str)(self.m_grad) )
        assert(self.channels == 3), "Wrong number of chanels"
        self.m_grad = m_grad
        assert(isinstance(self.m_grad, bool)), "Wrong type of m_grad for SeamlessCloningSolver class init "
        assert(self.height > 2 and self.width > 2), "Image too small"

        '''preprocess the mask'''
        self.mask_numpt = 0
        if (mask.shape[-1] == 3):
            self.mask_preprocess()

        print("mask_numpt: " + (str)(self.mask_numpt) )
        
        '''preprocess the source'''
        self.source_R = source[:,:,0]
        self.source_G = source[:,:,1]
        self.source_B = source[:,:,2]

        '''preprocess the target'''
        self.target_R = target[:,:,0]
        self.target_G = target[:,:,1]
        self.target_B = target[:,:,2]
        
        '''gradient_h : (height - 1, width -1)'''
        '''gradient_W : (height, width -1)'''
        self.grad_source_R_h = self.grad(self.source_R, self.height, self.width, "h")
        self.grad_source_R_w = self.grad(self.source_R, self.height, self.width, "w")
        self.grad_source_G_h = self.grad(self.source_G, self.height, self.width, "h")
        self.grad_source_G_w = self.grad(self.source_G, self.height, self.width, "w")
        self.grad_source_B_h = self.grad(self.source_B, self.height, self.width, "h")
        self.grad_source_B_w = self.grad(self.source_B, self.height, self.width, "w")
        self.grad_target_R_h = self.grad(self.target_R, self.height, self.width, "h")
        self.grad_target_R_w = self.grad(self.target_R, self.height, self.width, "w")
        self.grad_target_G_h = self.grad(self.target_G, self.height, self.width, "h")
        self.grad_target_G_w = self.grad(self.target_G, self.height, self.width, "w")
        self.grad_target_B_h = self.grad(self.target_B, self.height, self.width, "h")
        self.grad_target_B_w = self.grad(self.target_B, self.height, self.width, "w")

        '''generate the guiding field'''
        if (not m_grad):
            self.v_R_h = self.grad_source_R_h
            self.v_R_w = self.grad_source_R_w
            self.v_G_h = self.grad_source_G_h
            self.v_G_w = self.grad_source_G_w
            self.v_B_h = self.grad_source_B_h
            self.v_B_w = self.grad_source_B_w
        else:
            self.v_R_h = self.mix_grad(self.grad_source_R_h, self.grad_target_R_h)
            self.v_R_w = self.mix_grad(self.grad_source_R_w, self.grad_target_R_w)
            self.v_G_h = self.mix_grad(self.grad_source_G_h, self.grad_target_G_h)
            self.v_G_w = self.mix_grad(self.grad_source_G_w, self.grad_target_G_w)
            self.v_B_h = self.mix_grad(self.grad_source_B_h, self.grad_target_B_h)
            self.v_B_w = self.mix_grad(self.grad_source_B_w, self.grad_target_B_w)

    
    def grad(self, M, height, width, axis):
        if axis.lower() in {"w", "width", "wid"}:
            grad_ret = np.zeros((width, height-1), dtype=float)
            grad_ret = (M[:, 1:] - (M[:, :-1])) / 2
            return grad_ret
        elif  axis.lower() in {"h", "height"}:
            grad_ret = np.zeros((width, height-1), dtype=float)
            grad_ret = (M[1:, :] - (M[:-1, :])) / 2
            return grad_ret
        else:
            return np.array([])
    
    def mix_grad(self, v_1, v_2):
        assert(v_1.shape == v_2.shape), "mixed grads are of different shape"
        h, w = v_1.shape
        grad_ret = np.zeros_like(v_1)
        for i in range(h):
            for j in range(w):
                grad_ret[i,j] = max(v_1[i,j], v_2[i,j])
        return grad_ret

    def mask_preprocess(self):
        mask_o = np.zeros((self.height, self.width), dtype=float)
        mask_numpt = 0
        '''filter the mask and boundary region, integer larger than 0 denotes mask and 0.5 denotes boundary'''
        for i in range(self.height):
            for j in range(self.width):
                if all(self.mask[i, j] == [255, 255, 255]):
                    mask_numpt += 1
                    mask_o[i,j] = mask_numpt
                    '''boundary'''
                    if (i > 0 and mask_o[i-1,j] < 1):
                        mask_o[i-1,j] = 0.5
                    if (i < self.height-1 and mask_o[i+1,j] < 1):
                        mask_o[i+1,j] = 0.5
                    if (j > 0 and mask_o[i,j-1] < 1):
                        mask_o[i,j-1] = 0.5
                    if (i < self.width-1 and mask_o[i,j+1] < 1):
                        mask_o[i,j+1] = 0.5
        self.mask = mask_o
        self.mask_numpt = mask_numpt
    
    def solve(self):
        print("Start solving seamless cloning question")
        assert(isinstance(self.v_R_h, np.ndarray)), "v_R_h not a ndarray"
        assert(isinstance(self.v_R_w, np.ndarray)), "v_R_w not a ndarray"
        assert(isinstance(self.v_G_h, np.ndarray)), "v_G_h not a ndarray"
        assert(isinstance(self.v_G_w, np.ndarray)), "v_G_w not a ndarray"
        assert(isinstance(self.v_B_h, np.ndarray)), "v_B_h not a ndarray"
        assert(isinstance(self.v_B_w, np.ndarray)), "v_B_w not a ndarray"

        R_solver = GuidedInterpolationSolver(target = self.target_R, 
                                             mask = self.mask, 
                                             mask_numpt = self.mask_numpt, 
                                             g_field_h = self.v_R_h, 
                                             g_field_w = self.v_R_w, 
                                             height = self.height, 
                                             width = self.width)
        G_solver = GuidedInterpolationSolver(self.target_G, self.mask, self.mask_numpt, self.v_G_h, self.v_G_w, self.height, self.width)
        B_solver = GuidedInterpolationSolver(self.target_B, self.mask, self.mask_numpt, self.v_B_h, self.v_B_w, self.height, self.width)
        print("Start solving GuidedInterpolationSolver subquestion for R channel")
        R_output = R_solver.solve()
        print("Start solving GuidedInterpolationSolver subquestion for G channel")
        G_output = G_solver.solve()
        print("Start solving GuidedInterpolationSolver subquestion for B channel")
        B_output = B_solver.solve()
        print("Solving Finished")
        return np.stack((R_output, G_output, B_output), axis = 2)

class TextureFlatteningSolver(object):
    def __init__(self, source, mask, f_threshold):
        self.source = source
        self.mask = mask
        self.f_threshold = f_threshold
        self.height, self.width, self.channels = source.shape
        print("height: " + (str)(self.height) )
        print("width: " + (str)(self.width) )
        assert(self.channels == 3), "Wrong number of chanels"
        assert(self.height > 2 and self.width > 2), "Image too small"

        '''preprocess the mask'''
        self.mask_numpt = 0
        if (mask.shape[-1] == 3):
            self.mask_preprocess()
        print("mask_numpt: " + (str)(self.mask_numpt) )

        '''preprocess the source'''
        self.source_R = source[:,:,0]
        self.source_G = source[:,:,1]
        self.source_B = source[:,:,2]

        '''gradient_h : (height - 1, width -1)'''
        '''gradient_W : (height, width -1)'''
        self.grad_source_R_h = self.grad(self.source_R, self.height, self.width, "h")
        self.grad_source_R_w = self.grad(self.source_R, self.height, self.width, "w")
        self.grad_source_G_h = self.grad(self.source_G, self.height, self.width, "h")
        self.grad_source_G_w = self.grad(self.source_G, self.height, self.width, "w")
        self.grad_source_B_h = self.grad(self.source_B, self.height, self.width, "h")
        self.grad_source_B_w = self.grad(self.source_B, self.height, self.width, "w")

        '''generate the guiding field'''
        v_R_h = self.grad_source_R_h
        v_R_h[v_R_h < f_threshold] = 0
        self.v_R_h = v_R_h
        v_R_w = self.grad_source_R_w
        v_R_w[v_R_w < f_threshold] = 0
        self.v_R_w = v_R_w

        v_G_h = self.grad_source_G_h
        v_G_h[v_G_h < f_threshold] = 0
        self.v_G_h = v_G_h
        v_G_w = self.grad_source_G_w
        v_G_w[v_G_w < f_threshold] = 0
        self.v_G_w = v_G_w

        v_B_h = self.grad_source_B_h
        v_B_h[v_B_h < f_threshold] = 0
        self.v_B_h = v_B_h
        v_B_w = self.grad_source_B_w
        v_B_w[v_B_w < f_threshold] = 0
        self.v_B_w = v_B_w
    
    def grad(self, M, height, width, axis):
        if axis.lower() in {"w", "width", "wid"}:
            grad_ret = np.zeros((width, height-1), dtype=float)
            grad_ret = (M[:, 1:] - (M[:, :-1])) / 2
            return grad_ret
        elif  axis.lower() in {"h", "height"}:
            grad_ret = np.zeros((width, height-1), dtype=float)
            grad_ret = (M[1:, :] - (M[:-1, :])) / 2
            return grad_ret
        else:
            return np.array([])
    
    def mask_preprocess(self):
        mask_o = np.zeros((self.height, self.width), dtype=float)
        mask_numpt = 0
        '''filter the mask and boundary region, integer larger than 0 denotes mask and 0.5 denotes boundary'''
        for i in range(self.height):
            for j in range(self.width):
                if all(self.mask[i, j] == [255, 255, 255]):
                    mask_numpt += 1
                    mask_o[i,j] = mask_numpt
                    '''boundary'''
                    if (i > 0 and mask_o[i-1,j] < 1):
                        mask_o[i-1,j] = 0.5
                    if (i < self.height-1 and mask_o[i+1,j] < 1):
                        mask_o[i+1,j] = 0.5
                    if (j > 0 and mask_o[i,j-1] < 1):
                        mask_o[i,j-1] = 0.5
                    if (i < self.width-1 and mask_o[i,j+1] < 1):
                        mask_o[i,j+1] = 0.5
        self.mask = mask_o
        self.mask_numpt = mask_numpt
    
    def solve(self):
        print("Start solving selection editing question")
        assert(isinstance(self.v_R_h, np.ndarray)), "v_R_h not a ndarray"
        assert(isinstance(self.v_R_w, np.ndarray)), "v_R_w not a ndarray"
        assert(isinstance(self.v_G_h, np.ndarray)), "v_G_h not a ndarray"
        assert(isinstance(self.v_G_w, np.ndarray)), "v_G_w not a ndarray"
        assert(isinstance(self.v_B_h, np.ndarray)), "v_B_h not a ndarray"
        assert(isinstance(self.v_B_w, np.ndarray)), "v_B_w not a ndarray"

        R_solver = GuidedInterpolationSolver(self.source_R, self.mask, self.mask_numpt, self.v_R_h, self.v_R_w, self.height, self.width)
        G_solver = GuidedInterpolationSolver(self.source_G, self.mask, self.mask_numpt, self.v_G_h, self.v_G_w, self.height, self.width)
        B_solver = GuidedInterpolationSolver(self.source_B, self.mask, self.mask_numpt, self.v_B_h, self.v_B_w, self.height, self.width)
        print("Start solving TextureFlatteningSolver subquestion for R channel")
        R_output = R_solver.solve()
        print("Start solving TextureFlatteningSolver subquestion for G channel")
        G_output = G_solver.solve()
        print("Start solving TextureFlatteningSolver subquestion for B channel")
        B_output = B_solver.solve()
        print("Solving Finished")
        return np.stack((R_output, G_output, B_output), axis = 2)

class IlluminationChangeSolver(object):
    def __init__(self, source, mask, alpha, beta):
        self.source = source
        self.mask = mask
        self.alpha = alpha
        self.beta = beta
        self.height, self.width, self.channels = source.shape
        print("height: " + (str)(self.height) )
        print("width: " + (str)(self.width) )
        assert(self.channels == 3), "Wrong number of chanels"
        assert(self.height > 2 and self.width > 2), "Image too small"

        '''preprocess the mask'''
        self.mask_numpt = 0
        if (mask.shape[-1] == 3):
            self.mask_preprocess()
        print("mask_numpt: " + (str)(self.mask_numpt))

        '''preprocess the source'''
        self.source_R = source[:,:,0]
        self.source_G = source[:,:,1]
        self.source_B = source[:,:,2]

        '''gradient_h : (height - 1, width)'''
        '''gradient_W : (height, width -1)'''
        self.grad_source_R_h = self.grad(self.source_R, self.height, self.width, "h")
        self.grad_source_R_w = self.grad(self.source_R, self.height, self.width, "w")
        self.grad_source_G_h = self.grad(self.source_G, self.height, self.width, "h")
        self.grad_source_G_w = self.grad(self.source_G, self.height, self.width, "w")
        self.grad_source_B_h = self.grad(self.source_B, self.height, self.width, "h")
        self.grad_source_B_w = self.grad(self.source_B, self.height, self.width, "w")

        '''Calculate the average'''
        self.grad_mean_R = (np.mean(np.abs(self.grad_source_R_h)) * (self.height - 1) * (self.width) + np.mean(np.abs(self.grad_source_R_w)) * (self.height) * (self.width-1)) / ((self.height - 1) * (self.width) + (self.height) * (self.width-1))
        self.grad_mean_G = (np.mean(np.abs(self.grad_source_G_h)) * (self.height - 1) * (self.width) + np.mean(np.abs(self.grad_source_G_w)) * (self.height) * (self.width-1)) / ((self.height - 1) * (self.width) + (self.height) * (self.width-1))
        self.grad_mean_B = (np.mean(np.abs(self.grad_source_B_h)) * (self.height - 1) * (self.width) + np.mean(np.abs(self.grad_source_B_w)) * (self.height) * (self.width-1)) / ((self.height - 1) * (self.width) + (self.height) * (self.width-1))

        '''Generate the guiding vector field'''
        self.v_R_h = self.v_generate(self.grad_mean_R, self.grad_source_R_h)
        self.v_R_w = self.v_generate(self.grad_mean_R, self.grad_source_R_w)
        self.v_G_h = self.v_generate(self.grad_mean_G, self.grad_source_R_h)
        self.v_G_w = self.v_generate(self.grad_mean_G, self.grad_source_R_w)
        self.v_B_h = self.v_generate(self.grad_mean_B, self.grad_source_R_h)
        self.v_B_w = self.v_generate(self.grad_mean_B, self.grad_source_R_w)
    
    def v_generate(self, average, grad_source):
        v_ret = (self.alpha * average)** self.beta * ((np.abs(grad_source) + 1e-12) ** (-self.beta)) * grad_source
        return v_ret

    def mask_preprocess(self):
        mask_o = np.zeros((self.height, self.width), dtype=float)
        mask_numpt = 0
        '''filter the mask and boundary region, integer larger than 0 denotes mask and 0.5 denotes boundary'''
        for i in range(self.height):
            for j in range(self.width):
                if all(self.mask[i, j] == [255, 255, 255]):
                    mask_numpt += 1
                    mask_o[i,j] = mask_numpt
                    '''boundary'''
                    if (i > 0 and mask_o[i-1,j] < 1):
                        mask_o[i-1,j] = 0.5
                    if (i < self.height-1 and mask_o[i+1,j] < 1):
                        mask_o[i+1,j] = 0.5
                    if (j > 0 and mask_o[i,j-1] < 1):
                        mask_o[i,j-1] = 0.5
                    if (i < self.width-1 and mask_o[i,j+1] < 1):
                        mask_o[i,j+1] = 0.5
        self.mask = mask_o
        self.mask_numpt = mask_numpt
    
    def grad(self, M, height, width, axis):
        if axis.lower() in {"w", "width", "wid"}:
            grad_ret = np.zeros((width, height-1), dtype=float)
            grad_ret = (M[:, 1:] - (M[:, :-1])) / 2
            return grad_ret
        elif  axis.lower() in {"h", "height"}:
            grad_ret = np.zeros((width, height-1), dtype=float)
            grad_ret = (M[1:, :] - (M[:-1, :])) / 2
            return grad_ret
        else:
            return np.array([])
        
    def solve(self):
        print("Start solving selection editing question")
        assert(isinstance(self.v_R_h, np.ndarray)), "v_R_h not a ndarray"
        assert(isinstance(self.v_R_w, np.ndarray)), "v_R_w not a ndarray"
        assert(isinstance(self.v_G_h, np.ndarray)), "v_G_h not a ndarray"
        assert(isinstance(self.v_G_w, np.ndarray)), "v_G_w not a ndarray"
        assert(isinstance(self.v_B_h, np.ndarray)), "v_B_h not a ndarray"
        assert(isinstance(self.v_B_w, np.ndarray)), "v_B_w not a ndarray"

        R_solver = GuidedInterpolationSolver(self.source_R, self.mask, self.mask_numpt, self.v_R_h, self.v_R_w, self.height, self.width)
        G_solver = GuidedInterpolationSolver(self.source_G, self.mask, self.mask_numpt, self.v_G_h, self.v_G_w, self.height, self.width)
        B_solver = GuidedInterpolationSolver(self.source_B, self.mask, self.mask_numpt, self.v_B_h, self.v_B_w, self.height, self.width)
        print("Start solving IlluminationChangeSolver subquestion for R channel")
        R_output = R_solver.solve()
        print("Start solving IlluminationChangeSolver subquestion for G channel")
        G_output = G_solver.solve()
        print("Start solving IlluminationChangeSolver subquestion for B channel")
        B_output = B_solver.solve()
        print("Solving Finished")
        return np.stack((R_output, G_output, B_output), axis = 2)

class ColorChangeSolver(object):
    def __init__(self, target, mask, R_mul, G_mul, B_mul):
        self.target = target
        self.mask = mask
        self.height, self.width, self.channels = source.shape
        print("height: " + (str)(self.height) )
        print("width: " + (str)(self.width) )
        assert(self.channels == 3), "Wrong number of chanels"
        assert(self.height > 2 and self.width > 2), "Image too small"

        '''preprocess the mask'''
        self.mask_numpt = 0
        if (mask.shape[-1] == 3):
            self.mask_preprocess()
        print("mask_numpt: " + (str)(self.mask_numpt))

        '''generate the source'''
        self.source = self.generate_source(R_mul, G_mul, B_mul)
        # cv2.imwrite("source_cc.jpg", self.source.astype(np.uint8))

        '''preprocess the source'''
        self.source_R = self.source[:,:,0]
        self.source_G = self.source[:,:,1]
        self.source_B = self.source[:,:,2]

        '''preprocess the target'''
        self.target_R = target[:,:,0]
        self.target_G = target[:,:,1]
        self.target_B = target[:,:,2]

        '''gradient_h : (height - 1, width)'''
        '''gradient_W : (height, width -1)'''
        self.grad_source_R_h = self.grad(self.source_R, self.height, self.width, "h")
        self.grad_source_R_w = self.grad(self.source_R, self.height, self.width, "w")
        self.grad_source_G_h = self.grad(self.source_G, self.height, self.width, "h")
        self.grad_source_G_w = self.grad(self.source_G, self.height, self.width, "w")
        self.grad_source_B_h = self.grad(self.source_B, self.height, self.width, "h")
        self.grad_source_B_w = self.grad(self.source_B, self.height, self.width, "w")
        self.grad_target_R_h = self.grad(self.target_R, self.height, self.width, "h")
        self.grad_target_R_w = self.grad(self.target_R, self.height, self.width, "w")
        self.grad_target_G_h = self.grad(self.target_G, self.height, self.width, "h")
        self.grad_target_G_w = self.grad(self.target_G, self.height, self.width, "w")
        self.grad_target_B_h = self.grad(self.target_B, self.height, self.width, "h")
        self.grad_target_B_w = self.grad(self.target_B, self.height, self.width, "w")

        '''Generate the guiding vector field'''
        self.v_R_h = self.grad_source_R_h * R_mul
        self.v_R_w = self.grad_source_R_w * R_mul
        self.v_G_h = self.grad_source_G_h * G_mul
        self.v_G_w = self.grad_source_G_w * G_mul
        self.v_B_h = self.grad_source_B_h * B_mul
        self.v_B_w = self.grad_source_B_w * B_mul

        # '''TT: Generate the guiding vector field'''
        # self.v_R_h_f = self.grad_target_R_h
        # self.v_R_w_f = self.grad_target_R_w
        # self.v_G_h_f = self.grad_target_R_h
        # self.v_G_w_f = self.grad_target_R_w
        # self.v_B_h_f = self.grad_target_R_h
        # self.v_B_w_f = self.grad_target_R_w

        # print(np.array_equal(self.v_R_h, self.v_R_h_f))
        # print(np.array_equal(self.v_G_h, self.v_G_h_f))
        # print(np.array_equal(self.v_B_h, self.v_B_h_f))
    
    def generate_source(self, R_mul: float, G_mul:float, B_mul:float):
        source_ret = self.target.copy()
        RGB_mul = np.array([R_mul, G_mul, B_mul],dtype=float)
        for i in range(self.height):
            for j in range(self.width):
                if (self.mask[i,j] >= 1):
                    source_ret[i,j] = self.target[i,j] * RGB_mul
        # cv2.imwrite("source_cc.jpg", source_ret.astype(np.uint8))
        return source_ret

    def mask_preprocess(self):
        mask_o = np.zeros((self.height, self.width), dtype=float)
        mask_numpt = 0
        '''filter the mask and boundary region, integer larger than 0 denotes mask and 0.5 denotes boundary'''
        for i in range(self.height):
            for j in range(self.width):
                if all(self.mask[i, j] == [255, 255, 255]):
                    mask_numpt += 1
                    mask_o[i,j] = mask_numpt
                    '''boundary'''
                    if (i > 0 and mask_o[i-1,j] < 1):
                        mask_o[i-1,j] = 0.5
                    if (i < self.height-1 and mask_o[i+1,j] < 1):
                        mask_o[i+1,j] = 0.5
                    if (j > 0 and mask_o[i,j-1] < 1):
                        mask_o[i,j-1] = 0.5
                    if (i < self.width-1 and mask_o[i,j+1] < 1):
                        mask_o[i,j+1] = 0.5
        self.mask = mask_o
        self.mask_numpt = mask_numpt
    
    def grad(self, M, height, width, axis):
        if axis.lower() in {"w", "width", "wid"}:
            grad_ret = np.zeros((width, height-1), dtype=float)
            grad_ret = (M[:, 1:] - (M[:, :-1])) / 2
            return grad_ret
        elif  axis.lower() in {"h", "height"}:
            grad_ret = np.zeros((width, height-1), dtype=float)
            grad_ret = (M[1:, :] - (M[:-1, :])) / 2
            return grad_ret
        else:
            return np.array([])
        
    def solve(self):
        print("Start selection editing cloning question")
        assert(isinstance(self.v_R_h, np.ndarray)), "v_R_h not a ndarray"
        assert(isinstance(self.v_R_w, np.ndarray)), "v_R_w not a ndarray"
        assert(isinstance(self.v_G_h, np.ndarray)), "v_G_h not a ndarray"
        assert(isinstance(self.v_G_w, np.ndarray)), "v_G_w not a ndarray"
        assert(isinstance(self.v_B_h, np.ndarray)), "v_B_h not a ndarray"
        assert(isinstance(self.v_B_w, np.ndarray)), "v_B_w not a ndarray"

        R_solver = GuidedInterpolationSolver(self.target_R, self.mask, self.mask_numpt, self.v_R_h, self.v_R_w, self.height, self.width)
        G_solver = GuidedInterpolationSolver(self.target_G, self.mask, self.mask_numpt, self.v_G_h, self.v_G_w, self.height, self.width)
        B_solver = GuidedInterpolationSolver(self.target_B, self.mask, self.mask_numpt, self.v_B_h, self.v_B_w, self.height, self.width)
        print("Start solving ColorChange subquestion for R channel")
        R_output = R_solver.solve()
        print("Start solving ColorChange subquestion for G channel")
        G_output = G_solver.solve()
        print("Start solving ColorChange subquestion for B channel")
        B_output = B_solver.solve()
        print("Solving Finished")
        return np.stack((R_output, G_output, B_output), axis = 2)

def str2bool(word):
    if isinstance(word, bool):
        return word
    if word.lower() in ('yes', 'true', 't', 'y', 'right', '1'):
        return True
    elif word.lower() in ('no', 'false', 'f', 'n', 'wrong', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Word with boolean meaning expected for argument mixing_gradients")

if __name__ == '__main__':
    '''
    This program executes poisson image editing tasks. The tasks can be divided to seamless cloning and selection editing

    For seamless cloning the user should:
    - state the task to be "seamless_clone" 
    - provide the path of the source image 
    - provide the path of the target image
    - provide the path of the mask image
    - provide the path for the output image
    - define whether to use mixing gradients or not 

    For selection editing the user should 
    - state the task to be "selection_editing" 
    - state the subtask between "texture_flattening", "illumination_change", "color_change", "tiling"
    - provide the path of the target image
    - provide the path of the mask image
    - provide the path for the output image
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help='Select task between seamless_clone and selection_editing', default="seamless_clone")
    parser.add_argument("--subtask", type=str, help='Select the subtask for selection editing in texture_flattening, illumination_change, color_change and tiling', default="texture_flattening")
    parser.add_argument("--source", type=str, help='the path of the source image, for example: img/archive2/6/source.png', default='img/archive2/6/source.png')
    parser.add_argument("--target", type=str, help='the path of the target image, for example: img/archive2/6/target.png', default='img/archive2/6/target.png')
    parser.add_argument("--mask", type=str, help='the path of the mask image, for example: img/archive2/6/mask.png', default='img/archive2/6/mask.png')
    parser.add_argument("--output", type=str, help='the path of the output image, for example: img/archive2/6/output.png', default='img/archive2/6/output.png')
    parser.add_argument("--mixing_gradients", type=str, help='whether use mixing gradients or not, type True or False', default="True")
    parser.add_argument("--flattening_threshold", type=float, help="The threshold for texutre flattening", default = 50)
    parser.add_argument("--illu_alpha", type=float, help="The value of the alpha used in illumination change", default = 0.2)
    parser.add_argument("--illu_beta", type=float, help="The value of the beta used in illumination change", default = 0.2)
    parser.add_argument("--R_mul", type=float, help="The parameter multiplied to the R channel in ColorChange", default = 0.5)
    parser.add_argument("--G_mul", type=float, help="The parameter multiplied to the G channel in ColorChange", default = 1.5)
    parser.add_argument("--B_mul", type=float, help="The parameter multiplied to the B channel in ColorChange", default = 0.5)
    args = parser.parse_args()
    task = args.task
    assert(task == 'seamless_clone' or task == 'selection_editing'), "Entered task unsupported"

    if (task == 'seamless_clone'):
        source = cv2.imread(args.source).astype(np.int32)
        target = cv2.imread(args.target).astype(np.int32)
        mask = cv2.imread(args.mask).astype(np.int32)
        output_pth = args.output
        m_grad = str2bool(args.mixing_gradients)
        assert (source.shape[:2] == target.shape[:2] and target.shape[:2] == mask.shape[:2]), "target, source and mask image are not in the same size"
        solver = SeamlessCloningSolver(source, target, mask, m_grad)
        output_img = solver.solve().astype(np.uint8)
        # print("output shape: " + (str)(output_img.shape))
        cv2.imwrite(output_pth, output_img)
        print("Output have been saved to " + output_pth)
    else:
        source = cv2.imread(args.source).astype(np.int32)
        mask = cv2.imread(args.mask).astype(np.int32)
        output_pth = args.output
        subtask = args.subtask
        if (subtask == "texture_flattening"):
            f_threshold = args.flattening_threshold
            solver = TextureFlatteningSolver(source, mask, f_threshold)
            output_img = solver.solve().astype(np.uint8)
            cv2.imwrite(output_pth, output_img)
            print("Output have been saved to " + output_pth)
        elif (subtask == "illumination_change"):
            illu_alpha = args.illu_alpha
            illu_beta = args.illu_beta
            solver = IlluminationChangeSolver(source, mask, illu_alpha, illu_beta)
            output_img = solver.solve().astype(np.uint8)
            cv2.imwrite(output_pth, output_img)
            print("Output have been saved to " + output_pth)
        elif (subtask == "color_change"):
            R_mul = args.R_mul
            G_mul = args.G_mul
            B_mul = args.B_mul
            solver = ColorChangeSolver(source, mask, R_mul, G_mul, B_mul)
            output_img = solver.solve().astype(np.uint8)
            cv2.imwrite(output_pth, output_img)
            print("Output have been saved to " + output_pth)

