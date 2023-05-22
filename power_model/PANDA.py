from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise 
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import math
import random
#import torch
#import torch.nn as nn
#import torch.nn.functional as f
import matplotlib.pyplot as plt

class McPAT_Calib_with_arch_knowledge:
    def __init__(self):
        self.microarch_dataset = np.load('feature_set_5_9.npy')
        self.eda_dataset = np.load('label_set_5_7.npy')
        #num_of_cycle:0 slack:1 area:2 leakage:3 dynamic:4 total:5 area_components:6-14 power_components:15-29
        self.microarch_dataset = self.microarch_dataset[:,1:,:]
        #McPAT:0-37 Params:38-51 gem5:52-162
        self.flatten_microarch = self.microarch_dataset.reshape(self.microarch_dataset.shape[0]*self.microarch_dataset.shape[1],self.microarch_dataset.shape[2])
        self.flatten_eda = self.eda_dataset.reshape(self.eda_dataset.shape[0]*self.eda_dataset.shape[1],self.eda_dataset.shape[2])
        self.event_feature_of_components={
            "DCache":[139,149],
            "ICache":[149,155],
            #"BTB":[155,159],
            "BP":[159,163],
            "RNU":[163,170],
            "Itlb":[170,172],
            "Dtlb":[172,174],
            "Regfile":[174,179],
            "ROB":[179,181],
            "IFU":[181,198],
            "LSU":[198,201],
            "FU_Pool":[201,203],
            "ISU":[203,206]
            
        }
        self.params_feature_of_components={
            "DCache":[8,10,11,12],
            "ICache":[10,13],
            #"BTB":[0,7],
            "BP":[0,7],
            "RNU":[1],
            "Itlb":[1],
            "Dtlb":[11],
            "Regfile":[1,4,5],
            "ROB":[1,3],
            "IFU":[0,1,2,13],
            "LSU":[6,8],
            "FU_Pool":[8,9],
            "ISU":[1,8,9]
        }
        
        self.encode_table={
            "BP":[0],
            "ICache":[0,1],
            "DCache":[0,1],
            "ISU":[0],
            "Logic":[1],
            "IFU":[1],
            "ROB":[1],
            "Regfile":[1,2],
            "RNU":[0],
            "Dtlb":[0],
            "LSU":[0]
            
        }
        #self.component_model_selection=[13,11,0,13,0,1,1,1,5,1,13]
        self.component_model_selection=[0,0,0,0,0,0,13,13,0,13,0,13]
        #self.others_model_selection=13
        self.others_model_selection=13
        self.logic_bias = 0
        self.dtlb_bias = 0
    
    def compute_reserve_station_entries(self,decodewidth_init):
        decodewidth = int(decodewidth_init+0.01)
        isu_params = [
            # IQT_MEM.numEntries IQT_MEM.dispatchWidth
            # IQT_INT.numEntries IQT_INT.dispatchWidth
            # IQT_FP.numEntries IQT_FP.dispatchWidth
            [8, decodewidth, 8, decodewidth, 8, decodewidth],
            [12, decodewidth, 20, decodewidth, 16, decodewidth],
            [16, decodewidth, 32, decodewidth, 24, decodewidth],
            [24, decodewidth, 40, decodewidth, 32, decodewidth],
            [24, decodewidth, 40, decodewidth, 32, decodewidth]
            ]
        _isu_params = isu_params[decodewidth - 1]
        return _isu_params[0]+_isu_params[2]+_isu_params[4]
    
    def train_sub_model(self,model,feature_set,label_set):
        #train_data = train_data.tolist()
        #label_data = label_data.tolist()
        model.fit(feature_set,label_set)
        return model
    
    def estimate_bias_logic(self,feature,label):
        feature_list = [int(feature[item]+0.01) for item in range(feature.shape[0])]
        num_of_feature = len(set(feature_list))
        if num_of_feature<=2:
            self.logic_bias = 4
        else:
            reg = LinearRegression().fit(feature.reshape(feature.shape[0],1), label.reshape(label.shape[0],1))
            bias = reg.intercept_
            alpha = reg.coef_[0]
            self.logic_bias = bias / alpha
            #print(num_of_feature,self.logic_bias)        
        return
    
    def estimate_bias_dtlb(self,feature,label):
        feature_list = [int(feature[item]+0.01) for item in range(feature.shape[0])]
        num_of_feature = len(set(feature_list))
        if num_of_feature<=1:
            self.dtlb_bias = 8
        else:
            reg = LinearRegression().fit(feature.reshape(feature.shape[0],1), label.reshape(label.shape[0],1))
            bias = reg.intercept_
            alpha = reg.coef_[0]
            self.dtlb_bias = bias / alpha
            #print(num_of_feature,self.dtlb_bias)        
        return
    
    def encode_arch_knowledge(self,component_name,feature,label):
        
        #print(feature[:,encode_index].shape)
        #print(label.shape)
        if component_name=="BP" or component_name=="ICache" or component_name=="DCache" or component_name=="RNU" or component_name=="ROB" or component_name=="IFU" or component_name=="LSU":
            scale_factor = np.ones(label.shape)
            for i in range(len(self.encode_table[component_name])):
                encode_index = self.encode_table[component_name][i] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0]
                acc_feature = feature[:,encode_index]
                scale_factor = scale_factor * acc_feature
            encode_label = label / scale_factor
        elif component_name=="Regfile":
            scale_factor = np.zeros(label.shape)
            for i in range(len(self.encode_table[component_name])):
                encode_index = self.encode_table[component_name][i] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0]
                acc_feature = feature[:,encode_index]
                scale_factor = scale_factor + acc_feature
            encode_label = label / scale_factor
        elif component_name=="ISU":
            encode_index = self.encode_table[component_name][0] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0]
            decodewidth = feature[:,encode_index]
            reserve_station = np.array([self.compute_reserve_station_entries(decodewidth[i]) for i in range(decodewidth.shape[0])])
            encode_label = label / reserve_station
        elif component_name=="Logic":
            encode_index = self.encode_table[component_name][0]
            self.estimate_bias_logic(feature[:,encode_index],label)
            #encode_label = (label - self.logic_bias) / feature[:,encode_index]
            encode_label = label / (feature[:,encode_index] + self.logic_bias)
        elif component_name=="Dtlb":
            encode_index = self.encode_table[component_name][0] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0]
            self.estimate_bias_dtlb(feature[:,encode_index],label)
            #encode_label = (label - self.logic_bias) / feature[:,encode_index]
            encode_label = label / (feature[:,encode_index] + self.dtlb_bias)
        return encode_label
    
    def decode_arch_knowledge(self,component_name,feature,pred):
        #decode_index = self.encode_table[component_name] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0] + 1
        #print(feature[:,decode_index].shape)
        #print(pred.shape)
        if component_name=="BP" or component_name=="ICache" or component_name=="DCache" or component_name=="RNU" or component_name=="ROB" or component_name=="IFU" or component_name=="LSU":
            scale_factor = np.ones(pred.shape)
            for i in range(len(self.encode_table[component_name])):
                decode_index = self.encode_table[component_name][i] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0]
                acc_feature = feature[:,decode_index]
                scale_factor = scale_factor * acc_feature
            decode_pred = pred * scale_factor
        elif component_name=="Regfile":
            scale_factor = np.zeros(pred.shape)
            for i in range(len(self.encode_table[component_name])):
                decode_index = self.encode_table[component_name][i] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0]
                acc_feature = feature[:,decode_index]
                scale_factor = scale_factor + acc_feature
            decode_pred = pred * scale_factor
        elif component_name=="ISU":
            decode_index = self.encode_table[component_name][0] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0]
            decodewidth = feature[:,decode_index]
            reserve_station = np.array([self.compute_reserve_station_entries(decodewidth[i]) for i in range(decodewidth.shape[0])])
            decode_pred = pred * reserve_station
        elif component_name=="Logic":
            decode_index = self.encode_table[component_name][0]
            #decode_pred = pred * feature[:,decode_index] + self.logic_bias
            decode_pred = pred * (feature[:,decode_index] + self.logic_bias)
        elif component_name=="Dtlb":
            decode_index = self.encode_table[component_name][0] + self.event_feature_of_components[component_name][1] - self.event_feature_of_components[component_name][0]
            #decode_pred = pred * feature[:,decode_index] + self.logic_bias
            decode_pred = pred * (feature[:,decode_index] + self.dtlb_bias)
        return decode_pred
    
    def build_per_component_model(self, model, training_index, feature_index, label_index, component_name):
        component_feature = self.flatten_microarch[:,feature_index]
        component_feature = component_feature[training_index,:]
        component_label = self.flatten_eda[:,label_index]
        component_label = component_label[training_index]
        if component_name=="BP" or component_name=="ICache" or component_name=="DCache" or component_name=="ISU" or component_name=="RNU" or component_name=="ROB" or component_name=="IFU" or component_name=="Regfile" or component_name=="Dtlb" or component_name=="LSU":
            component_label = self.encode_arch_knowledge(component_name,component_feature,component_label)
        #print(component_feature)
        #print(component_label)
        
        return_model = self.train_sub_model(model,component_feature,component_label)
        #if component_name=="ISU":
        #    print(return_model.coef_)
        return return_model
    
    def test_per_component_model(self, model, testing_index, feature_index, label_index):
        component_feature = self.flatten_microarch[:,feature_index]
        component_feature = component_feature[testing_index,:]
        component_label = self.flatten_eda[:,label_index]
        component_label = component_label[testing_index]
        pred = model.predict(component_feature)
        r2 = r2_score(component_label,pred)
        mape = mean_absolute_percentage_error(component_label,pred)
        return r2, mape
    
    def pred_per_component_result(self, model, testing_index, feature_index, component_name):
        component_feature = self.flatten_microarch[:,feature_index]
        component_feature = component_feature[testing_index,:]
        #print(component_feature.shape)
        #if component_name=="ISU":
        #    print(component_feature)
        pred = model.predict(component_feature)
        #if component_name=="ISU":
        #    print(pred)
        if component_name=="BP" or component_name=="ICache" or component_name=="DCache" or component_name=="ISU" or component_name=="RNU" or component_name=="ROB" or component_name=="IFU" or component_name=="Regfile" or component_name=="Dtlb" or component_name=="LSU":
            pred = self.decode_arch_knowledge(component_name, component_feature, pred)
        return pred
    
    
    def build_all_component_level_model(self, training_index,model_selection_list):
        model_list = []
        iter = 0
        for keys in self.event_feature_of_components.keys():
            start = (self.event_feature_of_components[keys])[0]
            end = (self.event_feature_of_components[keys])[1]
            feature_index = [item for item in range(start,end)]
            #feature_index.append(203+len(self.event_feature_of_components)+iter)
            param_index = self.params_feature_of_components[keys]
            param_index = [item+38 for item in param_index]
            feature_index = feature_index + param_index
            label_index = 6+len(self.event_feature_of_components)+iter
            model_selection = model_selection_list[iter]
            iter = iter + 1
            if model_selection==0:
                power_model = xgb.XGBRegressor()
            elif model_selection==1:
                power_model = LinearRegression()
            elif model_selection==2:
                power_model = Lasso()
            elif model_selection==3:
                power_model = Ridge()
            elif model_selection==4:
                power_model = ElasticNet()
            elif model_selection==5:
                power_model = BayesianRidge()
            elif model_selection==6:
                power_model = GaussianProcessRegressor()
            elif model_selection==7:
                power_model = KNeighborsRegressor()
            elif model_selection==8:
                power_model = SVR(kernel='poly')
            elif model_selection==9:
                power_model = SVR()
            elif model_selection==10:
                power_model = DecisionTreeRegressor()
            elif model_selection==11:
                power_model = RandomForestRegressor()
            elif model_selection==12:
                power_model = AdaBoostRegressor()
            elif model_selection==13:
                power_model = GradientBoostingRegressor()
            elif model_selection==14:
                power_model = BaggingRegressor()
                
            return_model = self.build_per_component_model(power_model,training_index,feature_index,label_index,keys)
            model_list.append(return_model)
        return model_list
    
    def test_all_component_level_model(self, testing_index, model_list):
        iter = 0
        r2_list = []
        mape_list = []
        for keys in self.event_feature_of_components.keys():
            start = (self.event_feature_of_components[keys])[0]
            end = (self.event_feature_of_components[keys])[1]
            feature_index = [item for item in range(start,end)]
            #feature_index.append(203+len(self.event_feature_of_components)+iter)
            param_index = self.params_feature_of_components[keys]
            param_index = [item+38 for item in param_index]
            feature_index = feature_index + param_index
            label_index = 6+len(self.event_feature_of_components)+iter
            model = model_list[iter]
            r2, mape = self.test_per_component_model(model,testing_index,feature_index,label_index)
            r2_list.append(r2)
            mape_list.append(mape)
            iter = iter + 1
        return r2_list,mape_list
    
    def pred_all_component_level_result(self, testing_index, model_list):
        iter = 0
        pred_list = []
        for keys in self.event_feature_of_components.keys():
            start = (self.event_feature_of_components[keys])[0]
            end = (self.event_feature_of_components[keys])[1]
            feature_index = [item for item in range(start,end)]
            #feature_index.append(203+len(self.event_feature_of_components)+iter)
            param_index = self.params_feature_of_components[keys]
            param_index = [item+38 for item in param_index]
            feature_index = feature_index + param_index
            model = model_list[iter]
            pred = self.pred_per_component_result(model,testing_index,feature_index,keys)
            #print("pred")
            #print(pred.shape)
            pred_list.append(pred)
            iter = iter + 1
        #print("len")
        #print(len(pred_list))
        return pred_list
    
    def build_others_model(self, training_index, model_selection):
        feature = self.flatten_microarch[training_index,:]
        feature = feature[:,38:139]
        label_total = self.flatten_eda[training_index,:]
        label = label_total[:,4]
        for i in range(len(self.params_feature_of_components)):
            component_index = 6+len(self.event_feature_of_components)+i
            label_tmp = label_total[:,component_index]
            label = label - label_tmp
        
        label = self.encode_arch_knowledge("Logic",feature,label)
        
        #print(label.shape)
        if model_selection==0:
            power_model = xgb.XGBRegressor()
        elif model_selection==1:
            power_model = LinearRegression()
        elif model_selection==2:
            power_model = Lasso()
        elif model_selection==3:
            power_model = Ridge()
        elif model_selection==4:
            power_model = ElasticNet()
        elif model_selection==5:
            power_model = BayesianRidge()
        elif model_selection==6:
            power_model = GaussianProcessRegressor()
        elif model_selection==7:
            power_model = KNeighborsRegressor()
        elif model_selection==8:
            power_model = SVR(kernel='poly')
        elif model_selection==9:
            power_model = SVR()
        elif model_selection==10:
            power_model = DecisionTreeRegressor()
        elif model_selection==11:
            power_model = RandomForestRegressor()
        elif model_selection==12:
            power_model = AdaBoostRegressor()
        elif model_selection==13:
            power_model = GradientBoostingRegressor()
        elif model_selection==14:
            power_model = BaggingRegressor()
        power_model.fit(feature,label)
        return power_model
    
    def test_top_level_model(self, testing_index, model):
        feature_arch = self.flatten_microarch[testing_index,:]
        feature_arch = feature_arch[:,38:139]
        feature_eda = self.flatten_eda[testing_index,:]
        feature_eda = feature_eda[:,6+len(self.event_feature_of_components):]
        #print(feature_arch.shape)
        #print(feature_eda.shape)
        feature = np.concatenate((feature_arch,feature_eda),axis=1)
        #print(feature.shape)
        label = self.flatten_eda[testing_index,:]
        label = label[:,4]
        pred = model.predict(feature)
        r2 = r2_score(label,pred)
        mape = mean_absolute_percentage_error(label,pred)
        return r2, mape
    
    def test_whole_model(self, testing_index, model_top, model_bottom):
        feature = self.flatten_microarch[testing_index,:]
        feature = feature[:,38:139]
        others_pred = model_top.predict(feature)
        
        others_pred = self.decode_arch_knowledge("Logic", feature, others_pred)
        
        component_pred = self.pred_all_component_level_result(testing_index,model_bottom)
        #print("Other ",others_pred)
        label_total = self.flatten_eda[testing_index,:]
        label = label_total[:,4]
        for i in range(len(self.params_feature_of_components)):
            component_index = 6+len(self.event_feature_of_components)+i
            label_tmp = label_total[:,component_index]
            label = label - label_tmp
        #print("Others R2={}".format(r2_score(label,others_pred)))
        #print("Others MAPE={}%".format(mean_absolute_percentage_error(label,others_pred)*100))
        #print("Comp ",component_pred)
        
        for i in range(len(self.params_feature_of_components)):
            component_index = 6+len(self.event_feature_of_components)+i
            label_tmp = label_total[:,component_index]
            #print("{} R2={}".format(i,r2_score(label_tmp,component_pred[i])))
            #print("{} MAPE={}%".format(i,mean_absolute_percentage_error(label_tmp,component_pred[i])*100))

        pred = others_pred
        for i in range(len(component_pred)):
            pred = pred + component_pred[i]
        label = self.flatten_eda[testing_index,:]
        label = label[:,4]
        r2 = r2_score(label,pred)
        mape = mean_absolute_percentage_error(label,pred)
        return r2, mape, pred, label
    
    def shuffle_split_component_model(self):
        fold = 15
        shuffled_index = [i for i in range(self.flatten_microarch.shape[0])]
        random.shuffle(shuffled_index)
        r2_ave_list = [[0 for j in range(len(self.event_feature_of_components))] for i in range(15)]
        mape_ave_list = [[0 for j in range(len(self.event_feature_of_components))] for i in range(15)]
        for i in range(fold):
            test_size = len(shuffled_index)//fold
            #print(test_size)
            testing_set = shuffled_index[i*test_size:(i+1)*test_size]
            training_set = shuffled_index[0:i*test_size]+shuffled_index[(i+1)*test_size:len(shuffled_index)]
            #print(len(training_set))
            for j in range(15):
                model_list = self.build_all_component_level_model(training_set,j)
                r2_list, mape_list = self.test_all_component_level_model(testing_set,model_list)
                for item in range(len(self.event_feature_of_components)):
                    r2_ave_list[j][item] = r2_ave_list[j][item] + r2_list[item]
                    mape_ave_list[j][item] = mape_ave_list[j][item] + mape_list[item]
        model_r2 = [0 for i in range(len(self.event_feature_of_components))]
        model_mape = [0 for i in range(len(self.event_feature_of_components))]
        max_r2_values = [-10000 for i in range(len(self.event_feature_of_components))]
        min_mape_values = [10000 for i in range(len(self.event_feature_of_components))]
        for i in range(15):
            for j in range(len(self.event_feature_of_components)):
                r2_ave_list[i][j] = r2_ave_list[i][j] / fold
                mape_ave_list[i][j] = mape_ave_list[i][j] / fold
                if r2_ave_list[i][j]>max_r2_values[j]:
                    max_r2_values[j]=r2_ave_list[i][j]
                    model_r2[j]=i
                if mape_ave_list[i][j]<min_mape_values[j]:
                    min_mape_values[j]=mape_ave_list[i][j]
                    model_mape[j]=i
        print("For shuffle_split:")
        for i in range(len(self.event_feature_of_components)):
            print(i)
            print("R2 = {},{}".format(max_r2_values[i],model_r2[i]))
            print("MAPE = {}%,{}".format(min_mape_values[i] * 100,model_mape[i]))
                

    
    def shuffle_split_top_model(self):
        fold = 15
        shuffled_index = [i for i in range(self.flatten_microarch.shape[0])]
        random.shuffle(shuffled_index)
        r2_ave = 0
        mape_ave = 0
        for i in range(fold):
            test_size = len(shuffled_index)//fold
            #print(test_size)
            testing_set = shuffled_index[i*test_size:(i+1)*test_size]
            training_set = shuffled_index[0:i*test_size]+shuffled_index[(i+1)*test_size:len(shuffled_index)]
            #print(len(training_set))
            model = self.build_top_level_model(training_set)
            r2, mape = self.test_top_level_model(testing_set,model)
            r2_ave = r2_ave + r2
            mape_ave = mape_ave + mape
        r2_ave = r2_ave / fold
        mape_ave = mape_ave / fold

        print("Shuffle_split")
        print("R2 = {}".format(r2_ave))
        print("MAPE = {}%".format(mape_ave * 100))
        return
    
    def shuffle_split_whole_model(self):
        fold = 15
        shuffled_index = [i for i in range(self.flatten_microarch.shape[0])]
        random.shuffle(shuffled_index)
        r2_ave_list = [0 for i in range(15)]
        mape_ave_list = [0 for i in range(15)]
        for i in range(fold):
            test_size = len(shuffled_index)//fold
            testing_set = shuffled_index[i*test_size:(i+1)*test_size]
            training_set = shuffled_index[0:i*test_size]+shuffled_index[(i+1)*test_size:len(shuffled_index)]
            for j in range(15):
                model_top = self.build_top_level_model(training_set,j)
                model_bottom = self.build_all_component_level_model(training_set,j)
                r2, mape = self.test_whole_model(testing_set, model_top, model_bottom)
                r2_ave_list[j] = r2_ave_list[j] + r2
                mape_ave_list[j] = mape_ave_list[j] + mape
        for i in range(15):
            r2_ave_list[i] = r2_ave_list[i] / fold
            mape_ave_list[i] = mape_ave_list[i] / fold
            print("Shuffle_split_{}".format(i))
            print("R2 = {}".format(r2_ave_list[i]))
            print("MAPE = {}%".format(mape_ave_list[i] * 100))  
        return
    
    def unknown_domain(self):
        fold = 5
        test_size = 24
        r2_ave_list = [[0 for j in range(len(self.event_feature_of_components))] for i in range(15)]
        mape_ave_list = [[0 for j in range(len(self.event_feature_of_components))] for i in range(15)]
        for i in range(fold):
            testing_set = [item for item in range(i*test_size,(i+1)*test_size)]
            training_set = [item for item in range(0,i*test_size)] + [item for item in range((i+1)*test_size,self.flatten_microarch.shape[0])]
            #random.shuffle(training_set)
            for j in range(15):
                model_list = self.build_all_component_level_model(training_set,j)
                r2_list, mape_list = self.test_all_component_level_model(testing_set,model_list)
                for item in range(len(self.event_feature_of_components)):
                    r2_ave_list[j][item] = r2_ave_list[j][item] + r2_list[item]
                    mape_ave_list[j][item] = mape_ave_list[j][item] + mape_list[item]
        model_r2 = [0 for i in range(len(self.event_feature_of_components))]
        model_mape = [0 for i in range(len(self.event_feature_of_components))]
        max_r2_values = [-10000 for i in range(len(self.event_feature_of_components))]
        min_mape_values = [10000 for i in range(len(self.event_feature_of_components))]
        for i in range(15):
            for j in range(len(self.event_feature_of_components)):
                r2_ave_list[i][j] = r2_ave_list[i][j] / fold
                mape_ave_list[i][j] = mape_ave_list[i][j] / fold
                if r2_ave_list[i][j]>max_r2_values[j]:
                    max_r2_values[j]=r2_ave_list[i][j]
                    model_r2[j]=i
                if mape_ave_list[i][j]<min_mape_values[j]:
                    min_mape_values[j]=mape_ave_list[i][j]
                    model_mape[j]=i
        print("For unknown_domain:")
        for i in range(len(self.event_feature_of_components)):
            print(i)
            print("R2 = {},{}".format(max_r2_values[i],model_r2[i]))
            print("MAPE = {}%,{}".format(min_mape_values[i] * 100,model_mape[i]))
        
    def unknown_config(self):
        fold = self.microarch_dataset.shape[0]
        test_size = self.microarch_dataset.shape[1]
        full_index = [i for i in range(self.flatten_microarch.shape[0])]
        r2_ave_list = [[0 for j in range(len(self.event_feature_of_components))] for i in range(15)]
        mape_ave_list = [[0 for j in range(len(self.event_feature_of_components))] for i in range(15)]
        for i in range(fold):
            testing_set = full_index[i*test_size:(i+1)*test_size]
            training_set = full_index[0:i*test_size]+full_index[(i+1)*test_size:len(full_index)]
            for j in range(15):
                model_list = self.build_all_component_level_model(training_set,j)
                r2_list, mape_list = self.test_all_component_level_model(testing_set,model_list)
                for item in range(len(self.event_feature_of_components)):
                    r2_ave_list[j][item] = r2_ave_list[j][item] + r2_list[item]
                    mape_ave_list[j][item] = mape_ave_list[j][item] + mape_list[item]
        model_r2 = [0 for i in range(len(self.event_feature_of_components))]
        model_mape = [0 for i in range(len(self.event_feature_of_components))]
        max_r2_values = [-10000 for i in range(len(self.event_feature_of_components))]
        min_mape_values = [10000 for i in range(len(self.event_feature_of_components))]
        for i in range(15):
            for j in range(len(self.event_feature_of_components)):
                r2_ave_list[i][j] = r2_ave_list[i][j] / fold
                mape_ave_list[i][j] = mape_ave_list[i][j] / fold
                if r2_ave_list[i][j]>max_r2_values[j]:
                    max_r2_values[j]=r2_ave_list[i][j]
                    model_r2[j]=i
                if mape_ave_list[i][j]<min_mape_values[j]:
                    min_mape_values[j]=mape_ave_list[i][j]
                    model_mape[j]=i
        print("For unknown_config:")
        for i in range(len(self.event_feature_of_components)):
            print(i)
            print("R2 = {},{}".format(max_r2_values[i],model_r2[i]))
            print("MAPE = {}%,{}".format(min_mape_values[i] * 100,model_mape[i]))
    
    def store_result(self,name,list_name):
        np.save(name,np.array(list_name))
    
    def unknown_domain_total(self):
        fold = 5
        test_size = 24
        r2_ave = 0
        mape_ave = 0
        pred_list = []
        label_list = []
        for i in range(fold):
            testing_set = [item for item in range(i*test_size,(i+1)*test_size)]
            training_set = [item for item in range(0,i*test_size)] + [item for item in range((i+1)*test_size,self.flatten_microarch.shape[0])]
            model_others = self.build_others_model(training_set,self.others_model_selection)
            model_components = self.build_all_component_level_model(training_set,self.component_model_selection)
            '''r2, mape = self.test_whole_model(training_set, model_others, model_components)
            print(i)
            print("R2_train = {}".format(r2))
            print("MAPE_train = {}%".format(mape * 100))'''  
            r2, mape, pred, label = self.test_whole_model(testing_set, model_others, model_components)
            pred_list = pred_list + pred.tolist()
            label_list = label_list + label.tolist()
            #print(i)
            #print("R2 = {}".format(r2))
            #print("MAPE = {}%".format(mape * 100))
            r2_ave = r2_ave + r2
            mape_ave = mape_ave + mape
        r2_ave = r2_ave / fold
        mape_ave = mape_ave / fold
        
        
        plt.clf()
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        plt.figure(figsize=(6, 5))
        plt.plot([0.4,1.6],[0.4,1.6],color='silver')
        color_set = ['b','g','r','c','m','y','k','skyblue','olive','gray','coral','gold','peru','pink','cyan','']
        for i in range(15):
            x = label_list[i*8:(i+1)*8]
            y = pred_list[i*8:(i+1)*8]
            plt.scatter(x,y,marker='.',color=color_set[i],label="BoomConfig{}".format(i),alpha=0.5,s=160)
        plt.xlabel('Ground Truth (W)', fontsize=22)
        plt.ylabel('Prediction (W)', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        r2_report = r2_score(label_list,pred_list)
        r_report = np.corrcoef(label_list,pred_list)[1][0]
        mape_report = mean_absolute_percentage_error(label_list,pred_list)
        
        np.save("unknown_domain_power_pred.npy",np.array(pred_list))
        np.save("unknown_domain_power_label.npy",np.array(label_list))
        
        plt.text(0.4,1.4,"MAPE={:.2f}%\nR={:.2f}".format(mape_report*100,r_report),fontsize=20,bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='silver',lw=5 ,alpha=0.7))
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.5)
        plt.savefig("result_figure/dynamic_unknown_domain_our.jpg",dpi=200)
        print("Unknown_domain")
        print("R = {}".format(r_report))
        print("MAPE = {}%".format(mape_report * 100)) 
        return r_report, mape_report
    
    def unknown_config_total(self):
        fold = self.microarch_dataset.shape[0]
        test_size = self.microarch_dataset.shape[1]
        full_index = [i for i in range(self.flatten_microarch.shape[0])]
        r2_ave = 0
        mape_ave = 0
        pred_list = []
        label_list = []
        for i in range(fold):
            testing_set = full_index[i*test_size:(i+1)*test_size]
            training_set = full_index[0:i*test_size]+full_index[(i+1)*test_size:len(full_index)]
            model_others = self.build_others_model(training_set,self.others_model_selection)
            model_components = self.build_all_component_level_model(training_set,self.component_model_selection)
            '''r2, mape = self.test_whole_model(training_set, model_others, model_components)
            print(i)
            print("R2_train = {}".format(r2))
            print("MAPE_train = {}%".format(mape * 100))  '''
            r2, mape, pred, label = self.test_whole_model(testing_set, model_others, model_components)
            pred_list = pred_list + pred.tolist()
            label_list = label_list + label.tolist()
            #print(i)
            #print("R2 = {}".format(r2))
            #print("MAPE = {}%".format(mape * 100))  
            r2_ave = r2_ave + r2
            mape_ave = mape_ave + mape
        r2_ave = r2_ave / fold
        mape_ave = mape_ave / fold
        
        plt.clf()
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        plt.figure(figsize=(6, 5))
        plt.plot([0.4,1.6],[0.4,1.6],color='silver')
        color_set = ['b','g','r','c','m','y','k','skyblue','olive','gray','coral','gold','peru','pink','cyan','']
        for i in range(15):
            x = label_list[i*8:(i+1)*8]
            y = pred_list[i*8:(i+1)*8]
            plt.scatter(x,y,marker='.',color=color_set[i],label="BoomConfig{}".format(i),alpha=0.5,s=160)
        plt.xlabel('Ground Truth (W)', fontsize=22)
        plt.ylabel('Prediction (W)', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        r2_report = r2_score(label_list,pred_list)
        r_report = np.corrcoef(label_list,pred_list)[1][0]
        mape_report = mean_absolute_percentage_error(label_list,pred_list)
        plt.text(0.4,1.4,"MAPE={:.2f}%\nR={:.2f}".format(mape_report*100,r_report),fontsize=20,bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='silver',lw=5 ,alpha=0.7))
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.5)
        plt.savefig("result_figure/dynamic_unknown_config_our.jpg",dpi=200)
        
        
        print("Unknown_config")
        print("R = {}".format(r_report))
        print("MAPE = {}%".format(mape_report * 100))  
        return r_report, mape_report
    
    def unknown_benchmark_total(self):
        fold = self.microarch_dataset.shape[1]
        test_size = self.microarch_dataset.shape[0]
        pred_list = []
        label_list = []
        for i in range(fold):
            training_set = []
            testing_set = []
            for j in range(test_size):
                testing_set.append(j*fold+i)
                #print(j*(fold+1))
                training_set = training_set + [item for item in range(j*fold,j*fold+i)] + [item for item in range(j*fold+i+1,(j+1)*fold)]
            model_others = self.build_others_model(training_set,self.others_model_selection)
            model_components = self.build_all_component_level_model(training_set,self.component_model_selection)
            '''r2, mape = self.test_whole_model(training_set, model_others, model_components)
            print(i)
            print("R2_train = {}".format(r2))
            print("MAPE_train = {}%".format(mape * 100))  '''
            r2, mape, pred, label = self.test_whole_model(testing_set, model_others, model_components)
            pred_list = pred_list + pred.tolist()
            label_list = label_list + label.tolist()

        r2_report = r2_score(label_list,pred_list)
        r_report = np.corrcoef(label_list,pred_list)[1][0]
        mape_report = mean_absolute_percentage_error(label_list,pred_list)
        
        
        print("Unknown_bench")
        print("R = {}".format(r_report))
        print("MAPE = {}%".format(mape_report * 100))  
        return
    
    def unknown_n_config(self,unknown):
        fold = 15
        test_size = 8 * unknown
        pred_list = []
        label_list = []
        pred_acc_vector = np.zeros((15*8))
        
        for i in range(fold):
            start_point = 8*i
            end_point = start_point + test_size
            if end_point<=self.flatten_microarch.shape[0]:
                testing_set = [item for item in range(start_point,end_point)]
                training_set = [item for item in range(0,start_point)] + [item for item in range(end_point,self.flatten_microarch.shape[0])]
            else:
                end_point = end_point - self.flatten_microarch.shape[0]
                testing_set = [item for item in range(start_point,self.flatten_microarch.shape[0])] + [item for item in range(0,end_point)]
                training_set = [item for item in range(end_point,start_point)]
                
            #testing_set = [item for item in range(i*test_size,(i+1)*test_size)]
            #training_set = [item for item in range(0,i*test_size)] + [item for item in range((i+1)*test_size,self.flatten_microarch.shape[0])]
            #random.shuffle(training_set)
            model_others = self.build_others_model(training_set,self.others_model_selection)
            model_components = self.build_all_component_level_model(training_set,self.component_model_selection)
            r2, mape, pred, label = self.test_whole_model(testing_set, model_others, model_components)
            #label = self.flatten_eda[testing_set,4]
            pred_list = pred_list + pred.tolist()
            label_list = label_list + label.tolist()
            pred_acc_vector[testing_set] = pred_acc_vector[testing_set] + pred
        
        
        r_report = np.corrcoef(label_list,pred_list)[1][0]
        mape_report = mean_absolute_percentage_error(label_list,pred_list)
        print("Unknown_{}_config".format(unknown))
        print("R = {}".format(r_report))
        print("MAPE = {}%".format(mape_report * 100))  
        
        pred_acc_vector = pred_acc_vector / unknown
        label = self.flatten_eda[:,4]
        
        #
        
        plt.clf()
        
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})
        plt.figure(figsize=(6, 5))
        
        
        plt.plot([0.4,1.6],[0.4,1.6],color='silver')
        color_set = ['b','g','r','c','m','y','k','skyblue','olive','gray','coral','gold','peru','pink','cyan','']
        for i in range(15):
            x = label[i*8:(i+1)*8]
            y = pred_acc_vector[i*8:(i+1)*8]
            plt.scatter(x,y,marker='.',color=color_set[i],label="BoomConfig{}".format(i),alpha=0.5,s=160)

        plt.xlabel('Ground Truth (W)', fontsize=22)
        plt.ylabel('Prediction (W)', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
            
        r_report = np.corrcoef(label,pred_acc_vector)[1][0]
        mape_report = mean_absolute_percentage_error(label,pred_acc_vector)
        print("Unknown_{}_config".format(unknown))
        print("R = {}".format(r_report))
        print("MAPE = {}%".format(mape_report * 100))  
        
        plt.text(0.4,1.4,"MAPE={:.2f}%\nR={:.2f}".format(mape_report*100,r_report),fontsize=20,bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='silver',lw=5 ,alpha=0.7))
        ax = plt.gca()
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.5)
        plt.savefig("result_figure/dynamic_unknown_{}_arch_model_without_mcpat.jpg".format(unknown),dpi=200)
        
        
        return r_report, mape_report
        
Calibration = McPAT_Calib_with_arch_knowledge()

r0,mape0 = Calibration.unknown_config_total()
r1,mape1 = Calibration.unknown_domain_total()
r2,mape2 = Calibration.unknown_n_config(5)
r3,mape3 = Calibration.unknown_n_config(10)
r4,mape4 = Calibration.unknown_n_config(14)
np.save("our_table.npy",np.array([mape0,r0,mape1,r1,mape2,r2,mape3,r3,mape4,r4]))

curve_mape = []
curve_r = []
for i in range(1,15):
    r,mape = Calibration.unknown_n_config(i)
    curve_mape.append(mape)
    curve_r.append(r)
np.save("our_curve_mape.npy",np.array(curve_mape))
np.save("our_curve_r.npy",np.array(curve_r))

