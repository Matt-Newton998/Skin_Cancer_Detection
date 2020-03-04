### Animal Functions ###

## imports ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from PIL import Image

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, multilabel_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## Functons ##

# Image conversion functions

def convert_image_to_array(image):
    # Convert Image To Numpy array
    img_array = np.asarray(image)
    return img_array

def image_to_nparray(metadata, image_path, new_image_size=(224,224)):
    
    image_ids = metadata['image_id']
    image_array = []
    class_array = []
    
    for i in range(metadata.shape[0]):
        # Progress
        progress = round((i/metadata.shape[0])*100,0)
        if i%500==0: print(f"Progress:{progress}%")
        
        # Image import
        d = metadata.loc[i][1]
        img = Image.open(image_path + d + '.jpg')
        
        # Class Dictionary
        class_array.append(metadata.loc[i][8])
        
        # Image to array
        img = img.resize(new_image_size)
        img_array = convert_image_to_array(img)
        image_array.append(img_array)
    
#     data = pd.DataFrame(image_array, index=image_ids)
#     data['target'] = class_array
    
    return image_array, class_array
        

# Tensor Conversion Functions
def probs_tensor_to_list(tensor):
    """Converts tensors to Lists"""
    new_list = []
    for sample in tensor:
        temp_list = []
        
        for row in sample:
            sub_temp_list = []
            
            for i in row:
                sub_temp_list.append(float(i))
            
            temp_list.append(sub_temp_list)
            
        new_list.append(temp_list)
    return new_list


def tensor_to_list(tensor):
    new_list = []
    for row in tensor:
        for i in row:
            new_list.append(int(i))
    return new_list




# Model Evaluation Function

def roc_plot(model, y_train, train_prob, y_val, val_prob):
    """ Plot the roc curve for model"""
    train_prob = model.predict_proba()
    
    
    plt.figure(figsize=(7,7))
    for data in [[y_train, train_prob],[y_val, val_prob]]: # ,[y_test, test_prob]
        fpr, tpr, threshold = roc_curve(data[0], data[1])
        plt.plot(fpr, tpr)
    annot(fpr, tpr, threshold)
    
    #Getting the best threshold
    threshold_chosen = 0
    difference = 0
    for i in range(len(threshold)):
        temp = tpr[i]-fpr[i]
        if temp>difference:
            difference=temp
            threshold_chosen=threshold[i]
    threshold_chosen = round(threshold_chosen,2)
    print('Best Threshold =',threshold_chosen)
    
    #Plot the ROC_Curve
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.ylabel('TPR (power)')
    plt.xlabel('FPR (alpha)')
    plt.legend(['train','val'])
    plt.show()

def annot(fpr,tpr,thr):
    """Add annotations to the roc curve"""
    k=0
    for i,j in zip(fpr,tpr):
        if k %50 == 0:
            plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
        k+=1  

def plot_confusion_matrix(y_val, y_pred, classes, model_name=None):
    
    """Plots the confusion matrix with a clasifications report and the metrics of the number of missed frauds
    and the total frauds that are flagged"""
    
    print('\n clasification report:\n', classification_report(y_val, y_pred, target_names=classes))
    print("-----------------------------------------------")

    cnf_matrix = confusion_matrix(y_val, y_pred)
    
    # Create the basic matrix
    plt.imshow(cnf_matrix,  cmap=plt.cm.Purples) 

    # Add title and axis labels
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add appropriate axis scales
    class_names = ['',classes[0],'',classes[1],'']# set(y) # Get class labels to add to matrix
    tick_marks = [-0.5,0,0.5,1,1.5]
    
    # Add appropriate axis scales
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add labels to each cell
    thresh = cnf_matrix.max() / 2. # Used for text coloring below
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
            plt.text(j, i, cnf_matrix[i, j],
                     horizontalalignment='center',
                     color='white' if cnf_matrix[i, j] > thresh else 'black')
    
    # Add a legend
    plt.colorbar()
    plt.show()
        
        
        
def plot_multiclass_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = list(range(-1,len(classes) + 1))
    classes = [''] + classes + [''] 
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Plot the Precision & Recall of the model over the full range of thresholds
    """
    # Fix lengths
    precisions = list(precisions[:-1])
    recalls = list(recalls[:-1])
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the Threshold", fontsize=16)
    plt.plot(thresholds, precisions, "b--", label="Precision")
    plt.plot(thresholds, recalls, "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Threshold")
    plt.legend(loc='best')
    plt.show()

