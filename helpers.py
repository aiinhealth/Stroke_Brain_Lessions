import matplotlib.pyplot as plt

from ipywidgets import interact
import numpy as np
import SimpleITK as sitk
import cv2
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, multilabel_confusion_matrix

IMG_SIZE = 75 #224
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
seed = 1337

def explore_3D_array(arr: np.ndarray, cmap: str = 'gray'):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. 
  The purpose of this function to visual inspect the 2D arrays in the image. 

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  def fn(SLICE):
    plt.figure(figsize=(7,7))
    plt.imshow(arr[SLICE, :, :], cmap=cmap)

  interact(fn, SLICE=(0, arr.shape[0]-1))


def explore_3D_array_comparison(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray'):
  """
  Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
  The purpose of this function to visual compare the 2D arrays after some transformation. 

  Args:
    arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
    arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform    
    cmap : Which color map use to plot the slices in matplotlib.pyplot
  """

  assert arr_after.shape == arr_before.shape

  def fn(SLICE):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10,10))

    ax1.set_title('Before', fontsize=15)
    ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)

    ax2.set_title('After', fontsize=15)
    ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)

    plt.tight_layout()
  
  interact(fn, SLICE=(0, arr_before.shape[0]-1))


def show_sitk_img_info(img: sitk.Image):
  """
  Given a sitk.Image instance prints the information about the MRI image contained.

  Args:
    img : instance of the sitk.Image to check out
  """
  pixel_type = img.GetPixelIDTypeAsString()
  origin = img.GetOrigin()
  dimensions = img.GetSize()
  spacing = img.GetSpacing()
  direction = img.GetDirection()

  info = {'Pixel Type' : pixel_type, 'Dimensions': dimensions, 'Spacing': spacing, 'Origin': origin,  'Direction' : direction}
  for k,v in info.items():
    print(f' {k} : {v}')


def add_suffix_to_filename(filename: str, suffix:str) -> str:
  """
  Takes a NIfTI filename and appends a suffix.

  Args:
      filename : NIfTI filename
      suffix : suffix to append

  Returns:
      str : filename after append the suffix
  """
  if filename.endswith('.nii'):
      result = filename.replace('.nii', f'_{suffix}.nii')
      return result
  elif filename.endswith('.nii.gz'):
      result = filename.replace('.nii.gz', f'_{suffix}.nii.gz')
      return result
  else:
      raise RuntimeError('filename with unknown extension')


def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
  """Rescale an array linearly."""
  minimum, maximum = np.min(array), np.max(array)
  m = (new_max - new_min) / (maximum - minimum)
  b = new_min - m * minimum
  return m * array + b


def explore_3D_array_with_mask_contour(arr: np.ndarray, mask: np.ndarray, thickness: int = 1):
  """
  Given a 3D array with shape (Z,X,Y) This function will create an interactive
  widget to check out all the 2D arrays with shape (X,Y) inside the 3D array. The binary
  mask provided will be used to overlay contours of the region of interest over the 
  array. The purpose of this function is to visual inspect the region delimited by the mask.

  Args:
    arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
    mask : binary mask to obtain the region of interest
  """
  assert arr.shape == mask.shape
  
  _arr = rescale_linear(arr,0,1)
  _mask = rescale_linear(mask,0,1)
  _mask = _mask.astype(np.uint8)

  def fn(SLICE):
    arr_rgb = cv2.cvtColor(_arr[SLICE, :, :], cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(_mask[SLICE, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    arr_with_contours = cv2.drawContours(arr_rgb, contours, -1, (0,1,0), thickness)

    plt.figure(figsize=(7,7))
    plt.imshow(arr_with_contours)

  interact(fn, SLICE=(0, arr.shape[0]-1))

def create_testgenerator(dataset,preprocessing_function, image_size=IMG_SIZE, batch_size=1, class_mode='binary'):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rescale=1./255,
        fill_mode="constant",
        preprocessing_function=preprocessing_function
    )

    test_generator = test_datagen.flow_from_directory(
        dataset,
        class_mode=class_mode,
        shuffle=False,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed
    )

    return test_generator

def make_prediction(model, test_generator, class_mode='binary', verbose=False):
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    if verbose:
        print(predictions)
    if class_mode == 'binary':
        predictions = [1 if x > 0.5 else 0 for x in predictions]
    elif class_mode == 'categorical':
        predictions = np.argmax(predictions, axis=1)
    else:
        raise ValueError("Class mode must be in either 'binary' or 'categorical'")
    return predictions


import seaborn as sns
sns.set(font_scale=1)

def plot_conf_mat(y_test, y_preds, save_path=None):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    confmat = confusion_matrix(y_test, y_preds)
    
    confmat = confmat.astype('float')*100 / confmat.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(
        confmat,
        annot=True,
        cbar=True,
        fmt='.2f',
        vmin=0,
        vmax=100.0
    )
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    ax.xaxis.set_ticklabels(['Lacunar','MCA']); 
    ax.yaxis.set_ticklabels(['Lacunar','MCA']);

    # bottom, top = ax.get_ylim()
    # ax.set_ylim(bottom + 0.5, top - 0.5)

    if save_path != None:
        print(f"Plot saved in {save_path}")
        fig.savefig(save_path, bbox_inches = "tight")
        
        
        
        
def compute_metrics(y_true, y_pred):
    metrics = dict()

    acc = tf.keras.metrics.Accuracy()
    acc.update_state(y_true, y_pred)
    metrics['acc'] = acc.result().numpy()

    prec = tf.keras.metrics.Precision()
    prec.update_state(y_true, y_pred)
    metrics['precision'] = prec.result().numpy()

    rec = tf.keras.metrics.Recall()
    rec.update_state(y_true, y_pred)
    metrics['recall'] = rec.result().numpy()

    auc = tf.keras.metrics.AUC()
    auc.update_state(y_true, y_pred)
    metrics['auc'] = auc.result().numpy()

    sn = tf.keras.metrics.SensitivityAtSpecificity(0.5)
    sn.update_state(y_true, y_pred)
    metrics['SN'] = sn.result().numpy()

    sp = tf.keras.metrics.SpecificityAtSensitivity(0.5)
    sp.update_state(y_true, y_pred)
    metrics['SP'] = sp.result().numpy()
    
    return metrics
