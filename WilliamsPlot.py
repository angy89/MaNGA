import numpy
import matplotlib.pyplot as plt

# #%% Williams Plot functions
# def hat_matrix(X1): #Hat Matrix
#     #hat_mat= X1*numpy.invert((numpy.transpose(X2)*X2)) * numpy.transpose(X1)
#     hat_mat =  numpy.dot(numpy.dot(X1, numpy.linalg.inv(numpy.dot(X1.T, X1))), X1.T)
#     return hat_mat
# 
# def williams_plot(X_train, X_test, Y_true_train, Y_true_test, model, toPrint = True,toPlot=False,path = './',filename = ''):
#     #X_train=pd.DataFrame(X_train)
#     #X_test=pd.DataFrame(X_test)
#     #H_train2= hat_matrix(numpy.concatenate([X_train, X_test], axis=0))
#     
#     H_train= hat_matrix (X_train)
#     H_test= hat_matrix (X_test)
#     y_pred_train= model.predict(X_train)
#     y_pred_test= model.predict(X_test)
#     
#     y_pred_test = y_pred_test.reshape(y_pred_test.shape[0],)
#     y_pred_train = y_pred_train.reshape(y_pred_train.shape[0],)
#     Y_true_train = Y_true_train.reshape(Y_true_train.shape[0],)
#     Y_true_test = Y_true_test.reshape(Y_true_test.shape[0],)
#     
#     residual_train= numpy.abs(Y_true_train - y_pred_train)
#     residual_test= numpy.abs(Y_true_test - y_pred_test)
#     s_residual_train = ((residual_train) - numpy.mean(residual_train)) / numpy.std(residual_train)
#     s_residual_test = (residual_test - numpy.mean(residual_test))/ numpy.std(residual_test)
# 
#     #leverage_train = numpy.diag(H_train.as_matrix)
#     #leverage_test = numpy.diag(H_test.as_matrix)
#     
#     # leverage= numpy.diag(H_train2)
#     # leverage_train = leverage[0:X_train.shape[0]]
#     # leverage_test = leverage[X_train.shape[0]:]
#     leverage_train = numpy.diag(H_train)
#     leverage_test = numpy.diag(H_test)
# 
#     #y_lim= [min(leverage), max(leverage)]
#     #x_lim= [min(numpy.concatenate(s_residual_train,s_residual_test)),max(numpy.concatenate(s_residual_train, s_residual_test))]
#     #residual = numpy.append(s_residual_train, s_residual_test, axis=0)
#     p = X_train.shape[1] #features
#     n = X_train.shape[0] #training compounds
#     h_star = (3 * (p+1))/float(n)
# 
#     train_points_in_ad = 0
#     train_points_out_ad = 0
#     for i,j in list(zip(leverage_train, s_residual_train)):
#         if ((i <= h_star) and (abs(j) <= 3)):
#             train_points_in_ad = train_points_in_ad + 1
#         else:
#             train_points_out_ad = train_points_out_ad +1
#             
#     train_points_in_ad = train_points_in_ad*100/ leverage_train.shape[0] #compute relative percentage
#     train_points_out_ad = train_points_out_ad*100 / leverage_train.shape[0]
#             
#     test_points_in_ad = 0 
#     test_points_out_ad = 0
#     for i,j in list(zip(leverage_test, s_residual_test)):
#         if (i <= h_star) and (abs(j) <= 3):
#             test_points_in_ad = test_points_in_ad+ 1
#         else:
#             test_points_out_ad = test_points_out_ad +1
#             
#     test_points_in_ad = float(test_points_in_ad) *100 / leverage_test.shape[0] #compute relative percetage
#     test_points_out_ad = float(test_points_out_ad)*100 / leverage_test.shape[0]
#     
# #    if toPrint:
# #      print("Percetege of train points inside AD: {}%".format(train_points_in_ad))
# #      print("Percetege of test points inside AD: {}%".format(test_points_in_ad))
# #      print("h*: {}".format(h_star))
#         
#     
#     if min(leverage_train) < min(leverage_test):
#         min_leverage_boundary = min(leverage_train)-1
#     else:
#         min_leverage_boundary = min(leverage_test)
#     if max(leverage_train) > max(leverage_test):
#         max_leverage_boundary = max(leverage_train)+1
#     else:
#         max_leverage_boundary = max(leverage_test)+1
#         
# #    if h_star > max_residual_boundary:
# #        max_residual_boundary = h_star+1
#         
#     if min(s_residual_train) < min(s_residual_test):
#         min_residual_boundary = min(s_residual_train)
#     else:
#         min_residual_boundary = min(s_residual_test)
#         
#     if min_residual_boundary > -3:
#         min_residual_boundary = -3-1
#     
#     if max(s_residual_train) > max(s_residual_test):
#         max_residual_boundary = max(s_residual_train)
#     else:
#         max_residual_boundary = max(s_residual_test)
#         
#     if max_residual_boundary < 3:
#         max_residual_boundary = 3+1
#     
#     if toPlot:
# #      print("toPlot is true")
#       plt.plot([h_star, h_star], [min_residual_boundary, max_residual_boundary], 'r-', label='h*')
#       plt.plot([min_leverage_boundary, max_leverage_boundary], [3,3], 'b-')
#       plt.plot([min_leverage_boundary, max_leverage_boundary], [-3,-3], 'b-') 
#       plt.plot(leverage_train.tolist(), s_residual_train.tolist(), 'o', label='train')
#       plt.plot(leverage_test.tolist(), s_residual_test.tolist(), '^', label = 'test')
#         ##plt.xlim(int(min_leverage_boundary),int(max_leverage_boundary))
#         ##plt.xlim(int(min_leverage_boundary),int(max_leverage_boundary))    
#         ##plt.ylim(int(min_residual_boundary), int(max_residual_boundary)+1)
#         ##plt.xlim(-1500, 1500)
#         ##plt.ylim(-2,4)
#       plt.xlabel('Leverages')
#       plt.ylabel('Std Residuals')
#       plt.legend(loc='upper left', shadow=True)
#       plt.savefig(path+'williams_plot_'+filename+'.png')
#       plt.close()
#       plt.show()
#       #with open(path+'williams_plot_results.txt', 'w') as f:
#       #    f.write("Percetege of train points inside AD: {}%".format(train_points_in_ad)+"\n")
#       #    f.write("Percentege of train points outside AD: {}".format(train_points_out_ad)+"\n")
#       #    f.write("Percetege of test points inside AD: {}%".format(test_points_in_ad)+"\n")
#       #    f.write("Percentege of test points outside AD: {}".format(test_points_out_ad)+"\n")
#       #    f.write("h*: {}".format(h_star))
#     return test_points_in_ad,train_points_in_ad


def hat_matrix(X1):#, X2): #Hat Matrix
    hat_mat =  numpy.dot(numpy.dot(X1, numpy.linalg.inv(numpy.dot(X1.T, X1))), X1.T)
    return hat_mat
    
def williams_plot(X_train, X_test, Y_true_train, Y_true_test, model, toPrint = True,toPlot=False,path = './',filename = ''):
    H_train= hat_matrix(numpy.concatenate([X_train, X_test], axis=0))#, numpy.concatenate([X_train, X_test], axis=0))
    y_pred_train= model.predict(X_train)
    y_pred_test= model.predict(X_test)
    
    y_pred_test = y_pred_test.reshape(y_pred_test.shape[0],)
    y_pred_train = y_pred_train.reshape(y_pred_train.shape[0],)
    Y_true_train = Y_true_train.reshape(Y_true_train.shape[0],)
    Y_true_test = Y_true_test.reshape(Y_true_test.shape[0],)
    
    residual_train= numpy.abs(Y_true_train - y_pred_train)
    residual_test= numpy.abs(Y_true_test - y_pred_test)
    s_residual_train = ((residual_train) - numpy.mean(residual_train)) / numpy.std(residual_train)
    s_residual_test = (residual_test - numpy.mean(residual_test))/ numpy.std(residual_test)

    leverage= numpy.diag(H_train)
    leverage_train = leverage[0:X_train.shape[0]]
    leverage_test = leverage[X_train.shape[0]:]
    p = X_train.shape[1] #features
    n = X_train.shape[0] #+ X_test.shape[0] #training compounds
    h_star = (3 * (p+1))/float(n)
    
    train_points_in_ad = float(100 * numpy.sum(numpy.asarray(leverage_train < h_star) & numpy.asarray(s_residual_train<3))) / len(leverage_train)
    test_points_in_ad = float(100 * numpy.sum(numpy.asarray(leverage_test < h_star) & numpy.asarray(s_residual_test<3))) / len(leverage_test)

    test_lev_out = numpy.sum(numpy.asarray(leverage_test > h_star))
    
    if toPrint:
      print("Percetege of train points inside AD: {}%".format(train_points_in_ad))
      print("Percetege of test points inside AD: {}%".format(test_points_in_ad))
      print("h*: {}".format(h_star))

    if toPlot:
      plt.plot(leverage_train.tolist(),s_residual_train.tolist(),'o', label='train')
      plt.plot(leverage_test.tolist(),s_residual_test.tolist(),'^', label = 'test')
      plt.axhline(y=3, color='r', linestyle='-')
      plt.axhline(y=-3, color='r', linestyle='-')
      plt.axvline(x=h_star, color='k', linestyle='--')
      plt.ylim(bottom=-6)
      plt.xlabel('Leverage')
      plt.ylabel('Standardized Residuals')
      plt.legend(loc='lower right', shadow=True)
      plt.savefig(path+'wp/williams_plot_'+filename+'.pdf')
      plt.close()
      plt.show()

    return test_points_in_ad,train_points_in_ad,test_lev_out,h_star,leverage_train,leverage_test,s_residual_train,s_residual_test

