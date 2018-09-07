import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

#creating svm class
class Support_Vector_Machine:
    def __init__(self,visualization=True):
        self.visualization= visualization
        self.colors={1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    #train
    def fit(self,data):
        self.data =data
        #{||w|| : {w,b}}
        opt_dict = {}

        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]

        all_data =[]

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        # support vctors will be yi[wi, yi+b] = 1
        step_sizes =[self.max_feature_value * 0.1,self.max_feature_value * 0.01,
                     # point of expense:
                     self.max_feature_value * 0.001]
        
        #extremely expensive
        b_range_multiple = 5
        # we dont need to take as small of steps with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            #we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value * b_range_multiple),self.max_feature_value * b_range_multiple, step * b_multiple):
                    for transformation in transforms:
                        #applying our defined transformation to w 
                        # which we aquired above
                        w_t = w * transformation
                        found_option = True # test it, set to true(innocent until proven guilty)
                        #weakest link in svm fundamentally
                        #smo attempts to fix this a bit
                        #yi(xi.w+b)>=1
                        #### add a break here later
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b)>=1:
                                    found_option = False
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)]=[w_t,b]
                if w[0]<0:
                    optimize =True
                    print('Optimized a step ')
                else:
                    #w = [5,5]
                    # step = 1
                    #w- step [4,4]
                    w= w-step
            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]

            self.w =  opt_choice[1]
            self.b =  opt_choice[1]

            latest_optimum = opt_choice[0][0] + step * 2






    def predict(self,data):
        # sign (x.w+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0],features[1], s=200, marker=*,c=self.colors[classification])
        
        return classification
    # only for vislization
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data.dict]

        #hyperplane x, w+b
        # v = x.w +b
        #ps =1 
        #neg = 1
        #dec = 0
        #plot data points, show support vector and decision boundary hyper plane
        def hyperplane(x,w,b,v):
            return (-w[0] * x-b +v / w[1])

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        #positive suport vector hyperplane(will be a scalar value), will be y
        psv1 = hyperplane(hype_x_min, self.w,self.b,1)
        psv2 = hyperplane(hype_x_max, self.w,self.b,1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])

        # (w.x+b) = -1
        #negative suport vector hyperplane(will be a scalar value), will be y
        nsv1 = hyperplane(hype_x_min, self.w,self.b,-1)
        nsv2 = hyperplane(hype_x_max, self.w,self.b,-1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])

         # (w.x+b) = 0
        #positive suport vector hyperplane(will be a scalar value), will be y
        db1 = hyperplane(hype_x_min, self.w,self.b,0)
        db2 = hyperplane(hype_x_max, self.w,self.b,0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])


        plt.show()






#create sample data dictionary
data_dict = {-1:np.array([[1,7],[2,8],[3,8],]),
            1:np.array([[5,1],[6,-1],[7,3],])}


svm = Support_Vector_Machine()
svm.fit(data=data_dict)
svm.visualize()