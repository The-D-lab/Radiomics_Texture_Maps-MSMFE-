import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from skimage import data
import pandas as pd
from itertools import product

def diagnol_2(matrix, k):
    h, w = matrix.shape[:2]
    
    k -= h-1

    #print(length)
    
    arr_x = np.zeros((h, w))
    arr_y = np.zeros((h, w))    
    for x in range(h):
        arr_x[x, :] = x
        
    for y in range(w):
        arr_y[:, y] = y
        
    x_diag = np.diag(arr_x, k).astype(np.uint16)
    y_diag = np.diag(arr_y, k).astype(np.uint16)
    
    pairs = [[x, y] for x, y in zip(x_diag, y_diag)]
    vec = np.array([matrix[pair[0], pair[1]] for n, pair in enumerate(pairs)])
    return vec

def get_sums(k):
    all_pairs = list(product(range(1,k), repeat=2))
    sums = [pair for pair in all_pairs if sum(pair) == k]
    return sums

def diagnol(matrix, k):
    #print(k)
    h, w = matrix.shape[:2]
    k+=2
    pairs = get_sums(k)
    #print(pairs)
    vec = np.array([matrix[pair[0]-1, pair[1]-1] for n, pair in enumerate(pairs) if max(pair) <= h])
    return vec

class MSMFE:
    def __init__(self, ref, imgs=None, vmin=0, vmax=255, nbit=8, ks=5,
                 features=['asm', 'contrast', 'dissimilarity', 'energy', 'entropy', 'homogeneity']):   
        
        ref = self.normalize(ref)
        if imgs is not None:
            self.keys = imgs.keys()
            for key in imgs.keys():
                imgs[key] = self.normalize(imgs[key])
        
        print('Creating GLCM(s) ...')
        self.vmin = vmin
        self.vmax = vmax
        self.nbit = nbit
        self.ks   = ks
        self.glcm_ref = self.fast_glcm(ref)
        self.glcm_imgs = {}
        if imgs is not None:
            for key in self.keys:
                print('\t... processing', key, imgs[key].shape)
                self.glcm_imgs[key] = self.fast_glcm(imgs[key])
        self.features = features
        self.error = {}
        self.feature_maps = {}
        self.feature_maps_ref = {}
        
        for feature in features:
            print('Processing feature map', feature, '...')
            self.feature_maps_ref[feature] = self.get_feature_map(ref, self.glcm_ref, feature)
            if imgs is not None:
                img_feature_maps = {}
                for key in self.keys:
                    if imgs is not None:
                        img_feature_maps[key] = self.get_feature_map(imgs[key], self.glcm_imgs[key], feature)
                self.feature_maps[feature] = img_feature_maps         
                
        self.imgs = imgs
        
    def get_names(self):
        names = list(self.keys) + ['_Reference']
        return names
        
    def normalize(self, img):
        img = (img - img.min())/(img.max()-img.min())
        img *= 255
        img = img.astype(np.uint8)
        return img
        
    def get_feature_maps(self):
        if self.imgs is not None:
            all_feature_maps = self.feature_maps
            for feature in self.features:
                temp = all_feature_maps[feature]
                temp['_Reference'] = self.feature_maps_ref[feature]
                all_feature_maps[feature] = temp
            return all_feature_maps
        else:
            return self.feature_maps_ref
            
    def get_error(self):
        error_df = pd.DataFrame(index=self.keys, columns=self.features)
        for feature in self.features:
            for key in self.keys:
                img = self.feature_maps[feature][key]
                ref = self.feature_maps_ref[feature]
                error = ((ref - img) ** 2).mean()
                error_df.at[key, feature] = error
        return error_df
    
    def get_saliency(self, feature):
        saliencies = []
        for key in self.keys:
            img = self.feature_maps[feature][key]
            ref = self.feature_maps_ref[feature]
            saliencies.append((ref - img) ** 2)
        saliencies = np.asarray(saliencies)
        return saliencies        
        
    def fast_glcm(self, img, kernel_size=5):
        mi, ma = self.vmin, self.vmax

        h,w = img.shape
    
        # digitize
        bins = np.linspace(mi, ma+1, self.nbit+1)
        gl1 = np.digitize(img, bins) - 1
        gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)
    
        # make glcm
        glcm = np.zeros((self.nbit, self.nbit, h, w), dtype=np.uint8)
        for i in range(self.nbit):
            for j in range(self.nbit):
                mask = ((gl1==i) & (gl2==j))
                glcm[i,j, mask] = 1
    
        kernel = np.ones((self.ks, self.ks), dtype=np.uint8)
        for i in range(self.nbit):
            for j in range(self.nbit):
                glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)
    
        glcm = glcm.astype(np.float32)
        return glcm
    
    def get_means(self, img, glcm):
        h,w = img.shape
    
        mean_i = np.zeros((h,w), dtype=np.float32)
        for i in range(self.nbit):
            for j in range(self.nbit):
                mean_i += glcm[i,j] * i / (self.nbit)**2
        
        mean_j = np.zeros((h,w), dtype=np.float32)
        for j in range(self.nbit):
            for i in range(self.nbit):
                mean_j += glcm[i,j] * j / (self.nbit)**2
        
        return mean_i, mean_j
    
    def get_stds(self, img, glcm):
        h,w = img.shape
        
        mean_i, mean_j = self.get_means(img, glcm)

        std_i = np.zeros((h,w), dtype=np.float32)
        for i in range(self.nbit):
            for j in range(self.nbit):
                std_i += (glcm[i,j] * i - mean_i)**2   
        std_i = np.sqrt(std_i)
        
        std_j = np.zeros((h,w), dtype=np.float32)
        for j in range(self.nbit):
            for i in range(self.nbit):
                std_j += (glcm[i,j] * j - mean_j)**2   
        std_j = np.sqrt(std_j)        
        
        return mean_i, mean_j, std_i, std_j
    
    def get_max(self, glcm):
            max_  = np.max(glcm, axis=(0,1))
            return(max_)
    
    def get_feature_map(self, img, glcm, feature):
        h,w = img.shape

        if feature == 'contrast':
            cont = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    cont += glcm[i,j] * (i-j)**2
            out = cont
    
        elif feature == 'dissimilarity':
            diss = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    diss += glcm[i,j] * np.abs(i-j)
            out = diss
        
        elif feature == 'homogeneity' or feature == 'IDM':
            homo = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    homo += glcm[i,j] / (1.+(i-j)**2)
            out = homo
    
        elif feature == 'asm' or feature == 'joint energy':
            asm = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    asm  += glcm[i,j]**2
            out = asm
    
        elif feature == 'energy':
            asm = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    asm  += glcm[i,j]**2    
            ene = np.sqrt(asm)
            out = ene
    
        elif feature == 'entropy':
            pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./self.ks**2
            ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
            out = ent
            
        elif feature == 'entropy_2':
            ene =  np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    small = .000000001
                    temp_glcm = glcm.copy()
                    temp_glcm[temp_glcm == 0] = small
                    log = np.log(temp_glcm[i, j])                   
                    ene += glcm[i, j] * log              
            out = -ene

        elif feature == 'autocorrelation':
            ac = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    ac  += glcm[i,j]*(i*j)
            out = ac
        
        elif feature == 'correlation':
            ac = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    ac  += glcm[i,j]*(i*j)
                    
            mean_i, mean_j, std_i, std_j = self.get_stds(img, glcm)            
            mean = mean_i * mean_j
            nominator   = ac - mean
            denominator = std_i * std_j
            out = nominator / denominator
            
        elif feature == 'sum_squares':
            ss = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    ss += glcm[i,j]*(i-np.mean(glcm)**2) 
            out = ss
        
        elif feature == 'sum_average':
            sa = np.zeros((h,w), dtype=np.float32)
            for k in range(self.nbit * 2):
                small = .000000001
                temp_glcm = glcm.copy()
                temp_glcm[temp_glcm == 0] = small
                sums = diagnol(temp_glcm, k).copy()
                sum  = np.sum(sums, 0).squeeze()
                sa += sum * k
            out = sa

        elif feature == 'sum_entropy':
            se =  np.zeros((h,w), dtype=np.float32)
            for k in range(self.nbit*2):
                small = .000000001
                temp_glcm = glcm.copy()
                temp_glcm[temp_glcm == 0] = small
                sums1 = diagnol(temp_glcm, k).copy()
                sums2 = diagnol(temp_glcm, k).copy()
                log = np.sum(np.log(sums2), 0)
                sum = np.sum(sums1, 0)
                #print(log.shape)
                se += -(sum.squeeze() * log.squeeze())       
            out = se

        elif feature == 'sum_variance':
            sa = np.zeros((h,w), dtype=np.float32)
            for k in range(self.nbit * 2):
                small = .000000001
                temp_glcm = glcm.copy()
                temp_glcm[temp_glcm == 0] = small
                sums = diagnol(temp_glcm, k).copy()
                sum  = np.sum(sums, 0).squeeze()
                sa += sum * k
            
            sv = np.zeros((h,w), dtype=np.float32)
            for k in range((self.nbit * 2) - 1):
                small = .000000001
                temp_glcm = glcm.copy()
                temp_glcm[temp_glcm == 0] = small
                sums = diagnol(temp_glcm, k).copy()
                sum  = np.sum(sums, 0).squeeze()
                sv += ((k - sa) ** 2) * sum
            
            out = sv
        
        elif feature == 'diff_average' or feature == 'difference_average':
            da = np.zeros((h,w), dtype=np.float32)
            for k in range((self.nbit * 2) - 1):
                small = .000000001
                temp_glcm = glcm.copy()
                temp_glcm[temp_glcm == 0] = small
                sums = diagnol_2(temp_glcm, k).copy()
                sum  = np.sum(sums, 0).squeeze()
                da += sum * k
            out = da

        elif feature == 'diff_entropy' or feature == 'difference_entropy':
            de =  np.zeros((h,w), dtype=np.float32)
            for k in range((self.nbit*2) - 1):
                small = .000000001
                temp_glcm = glcm.copy()
                temp_glcm[temp_glcm == 0] = small
                sums1 = diagnol_2(temp_glcm, k).copy()
                sums2 = diagnol_2(temp_glcm, k).copy()
                log = np.sum(np.log(sums2), 0)
                sum = np.sum(sums1, 0)
                #print(log.shape)
                de += -(sum.squeeze() * log.squeeze())       
            out = de
        
        elif feature == 'diff_variance' or feature == 'difference_variance':
            da = np.zeros((h,w), dtype=np.float32)
            for k in range((self.nbit * 2) - 1):
                small = .000000001
                temp_glcm = glcm.copy()
                temp_glcm[temp_glcm == 0] = small
                sums = diagnol_2(temp_glcm, k).copy()
                sum  = np.sum(sums, 0).squeeze()
                da += sum * k
            
            dv = np.zeros((h,w), dtype=np.float32)
            for k in range((self.nbit * 2) - 1):
                small = .000000001
                temp_glcm = glcm.copy()
                temp_glcm[temp_glcm == 0] = small
                sums = diagnol_2(temp_glcm, k).copy()
                sum  = np.sum(sums, 0).squeeze()
                dv += ((k - da) ** 2) * sum
            
            out = dv
            
        elif feature == 'joint_average':
            ja = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):            
                    ja += glcm[i, j] * i
            out = ja
            
        elif feature == 'max_prob' or feature == 'maximum_probability':
            max_  = np.max(glcm, axis=(0,1))
            out = max_
        
        elif feature == 'cluster_prominence':
            cp = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    cp += ((i+j - (2*np.mean(glcm))) ** 3) * glcm[i,j]
            return cp
        
        elif feature == 'cluster_shade':
            cp = np.zeros((h,w), dtype=np.float32)
            for i in range(self.nbit):
                for j in range(self.nbit):
                    cp += ((i+j - (2*np.mean(glcm))) ** 4) * glcm[i,j]
            return cp            
            
        else:
            out = np.zeros((512, 512))
            print('\n', feature, 'is not a feature!\n')
        
        return out

    
