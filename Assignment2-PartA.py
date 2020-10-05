import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def calculateGradient(m, x, y, w0, w1, w2, w3, w4, w5):
        dw_0 = 1.0 / m * sum([(2 * (w0 - y[i])) for i in range(m)]) 
        dw_1 = 1.0 / m * sum([(2 * ((w0 + w1 * x[i]) - y[i]) * x[i]) for i in range(m)])
        dw_2 = 1.0 / m * sum([(2 * ((w0 + w1 * x[i] + w2 * np.power(x[i], 2)) - y[i]) * (np.power(x[i], 2))) for i in range(m)])
        dw_3 = 1.0 / m * sum([(2 * ((w0 + w1 * x[i]+ w2 * np.power(x[i], 2)  + w3 * np.power(x[i], 3)) - y[i]) * (np.power(x[i], 3))) for i in range(m)])
        dw_4 = 1.0 / m * sum([(2 * ((w0 + w1 * x[i] + w2 * np.power(x[i], 2) + w3 * np.power(x[i], 3) + w4 * np.power(x[i], 4)) - y[i]) * (np.power(x[i], 4))) for i in range(m)])
        dw_5 = 1.0 / m * sum([(2 * ((w0 + w1 * x[i] + w2 * np.power(x[i], 2) + w3 * np.power(x[i], 3) + w4 * np.power(x[i], 4) + w5 * np.power(x[i], 5)) - y[i]) * (np.power(x[i], 5))) for i in range(m)])
        # print('calculateGradient Result : dw_0 : {}, dw_1: {}, dw_2: {}. dw_3: {}, dw_4: {}. dw_5: {}'.format(dw_0, dw_1, dw_2, dw_3, dw_4, dw_5))
        return dw_0,dw_1,dw_2,dw_3,dw_4,dw_5

def calculateNewDerivate(w, grad, gamma):
        w_new = w - (gamma * grad)
        return w_new

def summationPlotFunction(x, w1, w2, w3, w4, w5):
        m = x.shape[0]
        result1 = 1.0 / m * w1
        result2 = 1.0 / m * sum([(w1 + w2 * x[i]) for i in range(m)])
        result3 = 1.0 / m * sum([(w1 + w2 * x[i] + w3 * np.power(x[i], 2)) for i in range(m)])
        result4 = 1.0 / m * sum([(w1 + w2 * x[i] + w3 * np.power(x[i], 2) + w4 * np.power(x[i], 3)) for i in range(m)])
        result5 = 1.0 / m * sum([(w1 + w2 * x[i] + w3 * np.power(x[i], 2) + w4 * np.power(x[i], 3) + w5 * np.power(x[i], 4)) for i in range(m)])
        # result5 = 1.0 / m * sum([(w0 + w1 * x[i] + w2 * np.power(x[i], 2) + w3 * np.power(x[i], 3) + w4 * np.power(x[i], 4) + w5 * np.power(x[i], 5)) for i in range(m)])
        print('result1 = {} result2 = {} result3 = {} result4 = {} result5 = {}'.format(result1, result2, result3, result4, result5))
        return np.array([result1, result2, result3, result4, result5])

def gradient_descent(gamma, x, y, ep = 0.01, max_iter = 10000):
        converged = False
        iterations = 0
        m = x.shape[0] # number of samples
        # initial w_n values
        w0 = np.random.rand(x.shape[1])
        w1 = np.random.rand(x.shape[1])
        w2 = np.random.rand(x.shape[1])
        w3 = np.random.rand(x.shape[1])
        w4 = np.random.rand(x.shape[1])
        w5 = np.random.rand(x.shape[1])
        print('w0: {} w1: {} w2: {} w3: {}w4: {} w5: {}'.format(w0, w1,w2, w3, w4, w5))
        # total error, J
        J = 1.0 / m * sum([((w0 + w1 * x[i] + w2 * x[i] ** 2 + w3 * x[i] ** 3 + w4 * x[i] ** 4 + w5 * x[i] ** 5) - y[i]) ** 2 for i in range(m)])
        print('J: {}'.format(J))
        # Iterate Loop
        while not converged:
                grad0, grad1, grad2, grad3, grad4, grad5 = calculateGradient(m, x, y, w0, w1, w2, w3, w4, w5)

                # update
                w0_new = calculateNewDerivate(w0, grad0, gamma)
                w1_new = calculateNewDerivate(w1, grad1, gamma)
                w2_new = calculateNewDerivate(w2, grad2, gamma) 
                w3_new = calculateNewDerivate(w3, grad3, gamma)
                w4_new = calculateNewDerivate(w4, grad4, gamma)
                w5_new = calculateNewDerivate(w5, grad5, gamma)
        
                # update w
                w0 = w0_new
                w1 = w1_new
                w2 = w2_new
                w3 = w3_new
                w4 = w4_new
                w5 = w5_new

                # mean squared
                e = 1.0 / m * sum([((w0 + w1 * x[i] + w2 * x[i] ** 2 + w3 * x[i] ** 3 + w4 * x[i] ** 4 + w5 * x[i] ** 5) - y[i]) ** 2 for i in range(m)]) 
                print('abs{} <= ep:{},'.format(abs(J-e), ep))
                
                if abs(J-e) <= ep:
                        print('Converged, iterations: ', iterations, '!!!')
                        converged = True
        
                J = e   # update 
                iterations += 1  # update iterations
        
                if iterations == max_iter:
                        print ('Max interactions exceeded!')
                        converged = True

        return w0, w1, w2, w3, w4, w5

if __name__ == '__main__':
        print('Main Start...')
        x = np.array([-1.67245526, -2.36540279, -2.14724263,  1.40539096,  1.24297767,
                -1.71043904,  2.31579097,  2.40479939, -2.22112823])
        x = np.reshape(x, (9, 1))
        y = np.array([-18.56122168, -24.99658931, -24.41907817,  -2.688209,
                -1.54725306,  -19.18190097,   1.74117419,
                3.97703338, -24.80977847])
        print('x.shape = {} y.shape = {}'.format(x.shape, y.shape))
        print('x = {}, type(x): {} '.format(x, type(x)))
        
        gamma = 0.0001 # learning rate
        ep = 0.01 # convergence criteria

        # call gredient decent
        w_good0, w_good1, w_good2, w_good3, w_good4, w_good5 = gradient_descent(gamma, x, y, ep, max_iter = 1000)
        print('w_good0 = {} w_good1 = {} w_good2 = {} w_good3 = {} w_good4 = {}'.format(w_good0, w_good1, w_good2, w_good3, w_good4))

        # plot
        resultPlotFunctionArray = summationPlotFunction(x, w_good0, w_good1, w_good2, w_good3, w_good4)
        w_good_plot_array = np.array([w_good0, w_good1, w_good2, w_good3, w_good4])
        plt.plot(x,y,'ro') # red training samples
        plt.plot(w_good_plot_array,resultPlotFunctionArray,'bo') # blue color functionplot
        plt.show()
        print("Done!")
