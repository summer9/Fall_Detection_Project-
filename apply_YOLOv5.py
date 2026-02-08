"""
This file contains the pipelines to train or test a YOLOv5 model [#github]_. By default, the Fall Detection dataset
is used for this pipeline. The images are resized to fit the model's input size.

The main pipeline defines three different pipelines: training, validation, and testing. The main pipeline looks like this:

.. raw:: html
   :file: ../../diagrams/dl_object_detection_yolov5/main.html

First, the network is trained, and every 2nd epoch, the results are validated on a separate set. The model with the lowest loss
on the validation set is saved to yolov5.pth (early stopping). The training and validation loss are shown in a TensorBoard
session. During validation, one of the processed tiles for each minibatch is shown in TensorBoard.

The testing pipeline loads the model and estimates bounding boxes. The testing pipeline outputs mean Average Precision.
The results are shown in a TensorBoard session.

.. rubric:: Frequent Issues

**1: Use model.eval() for validation and testing**\n
**2: model.eval() should NOT be used in the EfficientDet pipelines**

.. rubric:: Footnotes
.. [#github] The source GitHub repository containing the YoloV5 model that is used here can be at https://github.com/ultralytics/yolov5
.. [#workdir] All information regarding the runs is saved in the working directory (trained model and TensorBoard)
"""
import os
import sys
import torch
import time

from common.elements.utils import get_pt_std, get_pt_mean
from functools import partial
from albumentations import Resize

from common.data.datatypes import (
    SampleContainer,
    BoundingBox,
    AnnotationDict,
)
from common.data.datasets_info import ABCDatasetInfo
from common.data.transforms import RemapLabels
from common.elements.utils import (
    get_tmp_dir,
    static_var,
    wait_forever,
    get_cicd_test_type,
    CICDTestType,
)
from elements.visualize import create_tb
from elements.calc_loss import calc_yolov5_loss_pt
from elements.calc_metrics import calc_mean_average_precision, calc_overall_precision_recall_f1score_prob_range
from elements.load_data import (
    load_image_ski,
    get_dataloader_object_detection,
    get_enabled_classes
)
from elements.load_model import (
    create_yolov5_pt,
    load_model_pt,
)
from elements.optimize import (
    get_adam_optimizer_pt,
    back_prop_pt,
)
from elements.predict import (
    predict_yolov5_pt,
    predict_yolov5_boxes_pt,
    decode_yolov5_boxes_pt,
)
from elements.save_model import save_model_pt
from elements.save_results import save_boxes_to_textfile
from elements.tune_params import (
    get_reduce_lr_on_plateau_pt,
    tune_learning_rate_pt,
)
from elements.visualize import (
    show_loss_tb,
    log_loss,
    show_image_and_boxes_tb,
    show_text_tb,
    show_metrics_graph_tb,
)
import xml.etree.ElementTree as ET
from common.data.loaders.filepath_loaders import get_file_paths

class FallDatasetInfo(ABCDatasetInfo):
    def __init__(self, images_dir, annotations_dir, mean, std, class_names):
        super().__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.mean = mean
        self.std = std
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    def _load_file_info(self):
        """
        Return a list of SampleContainer objects for each image with its annotation file path.
        """
        image_filepaths = get_file_paths(self.images_dir, ignore_dirs=True, file_extensions=('.jpg', '.png', '.jpeg'))
        annot_filepaths = get_file_paths(self.annotations_dir, ignore_dirs=True, file_extensions=('.xml',))

        if len(image_filepaths) != len(annot_filepaths):
            print(f"Warning: Mismatch between images ({len(image_filepaths)}) and annotations ({len(annot_filepaths)})")

        scs = []
        for image_fp, annot_fp in zip(image_filepaths, annot_filepaths):
            sc = SampleContainer()
            sc.image_fpath = image_fp
            sc.annotations_fpaths[BoundingBox] = annot_fp
            scs.append(sc)
        return scs

def _get_working_dir():
    return get_tmp_dir(os.path.splitext((os.path.basename(sys.modules[__name__].__file__)))[0])

@torch.no_grad()
def run_testing(test_ds_info: ABCDatasetInfo, preprocess_test: list, working_dir=_get_working_dir(), dev: str = "cuda:0", wait=True):
    """
    This method implements the testing pipeline and spins up a TensorBoard where the results are displayed.
    It shows the Precision, Recall, and F1-score for a single value of the IoU and class probability threshold.

    .. raw:: html
        :file: ../../diagrams/dl_object_detection_yolov5/test.html

    :param test_ds_info: DatasetInfo object containing the information (data directories, etc.) of the test data.
    :param preprocess_test: list of data preprocessing functions
    :param wait: wait for user input before closing TensorBoard
    :param working_dir: the current working directory where all information is stored (model weights, tensorboard, etc.)
    :param dev: cuda device
    """
    os.chdir(working_dir)

    test_dataloader = get_dataloader_object_detection(ds_info=test_ds_info, preprocessing=preprocess_test, batch_size=32)

    classes, num_classes = get_enabled_classes(test_dataloader, preprocess_test)

    # create tensorboard writer
    writer = create_tb("tensorboard_testing")

    # load_model element
    model = create_yolov5_pt(model_name="yolov5s", num_classes=num_classes, device=dev)
    load_model_pt(model, os.path.abspath(os.path.join("models", "yolov5.pth")))
    model.eval()

    result_boxes_per_image = {}
    target_boxes_per_image = {}

    batch: SampleContainer
    for i, batch in enumerate(test_dataloader):
        # load_data element (load input data and put on GPU)
        input_data, annotations = batch.image_data.get().to(dev), batch.annotations.get(BoundingBox).to(dev)

        # predict element
        predictions = predict_yolov5_boxes_pt(model, input_data)

        # iterate through all the images
        for batch_index, boxes in enumerate(predictions):
            bs, c, h, w = input_data.shape
            # Load original image
            idx = (i * test_dataloader.batch_size) + batch_index
            in_filename = test_dataloader.dataset[idx].image_fpath
            img = load_image_ski(in_filename)
            target_boxes = test_dataloader.dataset[idx].annotations.get(BoundingBox)

            scale_y = h / img.shape[0]
            scale_x = w / img.shape[1]

            # scale boxes with confidence threshold
            result_boxes = [[y1 / scale_y, x1 / scale_x, y2 / scale_y, x2 / scale_x, class_id, class_prob]
                            for y1, x1, y2, x2, class_id, class_prob in boxes if class_prob >= 0]

            # scale target boxes
            target_boxes_scaled = [[y1 / scale_y, x1 / scale_x, y2 / scale_y, x2 / scale_x, class_id]
                                   for y1, x1, y2, x2, class_id in target_boxes]

            # Debug prints
            print(f"Image {idx} - Predictions: {boxes if boxes.numel() > 0 else 'No prediction data'}")
            print(f"Image {idx} - Target boxes: {target_boxes_scaled}")

            # visualize element (show bounding boxes on image)
            show_image_and_boxes_tb(img, boxes_result=result_boxes, epoch=idx, name="result",
                                    writer=writer, class_names=classes)
            show_image_and_boxes_tb(img, boxes_target=target_boxes_scaled, epoch=idx, name="target",
                                    writer=writer, class_names=classes)

            # save results for validation
            result_boxes_per_image[in_filename] = result_boxes
            target_boxes_per_image[in_filename] = target_boxes_scaled

    # Calculate metrics for multiple probability thresholds
    metrics = calc_overall_precision_recall_f1score_prob_range(result_boxes_per_image=result_boxes_per_image,
                                                              target_boxes_per_image=target_boxes_per_image,
                                                              iou_t=0.1, num_classes=num_classes)

    show_metrics_graph_tb(metrics=metrics, writer=writer)

    # calculate mAP
    map = calc_mean_average_precision(result_boxes_per_image=result_boxes_per_image,
                                     target_boxes_per_image=target_boxes_per_image, iou_t=0.5)
    show_text_tb(name="mAP", text=str(round(map, 2)), writer=writer)

    # ----- PER-CLASS AP (console) -----
    aps = calc_mean_average_precision(result_boxes_per_image=result_boxes_per_image,
                                      target_boxes_per_image=target_boxes_per_image,
                                      iou_t=0.5,
                                      return_per_class=True)

    class_names = test_ds_info.class_names
    for c, ap in enumerate(aps):
        print(f"AP@0.5 {class_names[c]}: {round(ap * 100, 2)}%")
    # ----------------------------------

    # save_results element (save all the boxes to a textfile)
    save_boxes_to_textfile(result_boxes_per_image, "output_test/boxes_per_image.txt")

    if wait:
        wait_forever("Press stop or CTRL+C when finished examining the TensorBoard.")

    return result_boxes_per_image

@static_var(best_loss=9999)
@torch.no_grad()
def run_validation(valid_loader, model, epoch, working_dir=_get_working_dir(), dev: str = "cuda:0", writer=None):
    """
    Validate YoloV5. A graph of the loss function is displayed in a TensorBoard session. Also, the
    bounding boxes of a single tile are shown to visually monitor the results.
    The model with the lowest loss is saved in the working_dir under models/yolov5.pth.

    .. raw:: html
       :file: ../../diagrams/dl_object_detection_yolov5/validate.html

    :param valid_loader: the validation dataloader
    :param model: the trained model
    :param epoch: the current training epoch (used for administration)
    :param working_dir: the current working directory where all information is stored (model weights, tensorboard, etc.)
    :param dev: the cuda device to perform validation on
    :param writer: TensorBoard writer instance
    """
    os.chdir(working_dir)
    class_threshold = 0.5
    loss_sum, loss_cnt = 0, 0

    batch: SampleContainer
    for batch in valid_loader:
        # load_data element
        input_data, annotations = batch.image_data.get().to(dev), batch.annotations.get(BoundingBox).to(dev)

        # predict element
        predictions = predict_yolov5_pt(model, input_data)
        #print(predictions[0][:, :, 4].max())  # Print max confidence score
        if loss_cnt == 0:
            # visualize element (show results on the first image in the batch)
            boxes_result = decode_yolov5_boxes_pt(model, predictions, input_data)
            show_image_and_boxes_tb(input_data[0].cpu().numpy(), boxes_result[0],
                                    boxes_target=annotations[0].cpu().numpy(), epoch=epoch,
                                    normalize=True, dataformats="CHW", writer=writer)

        # calc_loss element
        loss = calc_yolov5_loss_pt(p=predictions, targets=annotations, model=model, image_height=608, image_width=608)

        loss_sum += loss.cpu().item()
        loss_cnt += 1

    avg_loss = loss_sum / loss_cnt
    print(f"Epoch {epoch + 1}: Validation Loss = {avg_loss:.4f}")

    # visualize element
    show_loss_tb(avg_loss, epoch, name="valid_loss", writer=writer)
    log_loss(avg_loss, epoch, name="valid_loss")

    # save_model element
    if avg_loss < run_validation.best_loss:
        filename = os.path.abspath(os.path.join("models", "yolov5.pth"))
        print(f"Saving best model to {filename}")
        save_model_pt(model, filename, loss=avg_loss, epoch=epoch)
        run_validation.best_loss = avg_loss

def run_training(train_ds_info: ABCDatasetInfo, valid_ds_info: ABCDatasetInfo, preprocess_train_val: list, num_epochs: int = 200, working_dir=_get_working_dir(), dev: str = "cuda:0"):
    """
    Train YoloV5 on the training set. TensorBoard will automatically be started to display training progress.
    Training is executed for 50 epochs.

    .. raw:: html
       :file: ../../diagrams/dl_object_detection_yolov5/train.html

    :param train_ds_info: DatasetInfo object containing the information (data directories, etc.) of the training data.
    :param valid_ds_info: DatasetInfo object containing the information (data directories, etc.) of the validation data.
    :param working_dir: the current working directory where all information is stored (model weights, tensorboard, etc.)
    :param num_epochs: the amount of epochs the model should train for
    :param preprocess_train_val: list of data preprocessing functions
    :param dev: the cuda device to train on
    """
    os.chdir(working_dir)

    train_loader = get_dataloader_object_detection(ds_info=train_ds_info,
                                                  preprocessing=preprocess_train_val,
                                                  batch_size=32,
                                                  shuffle=True)

    valid_loader = get_dataloader_object_detection(ds_info=valid_ds_info,
                                                  preprocessing=preprocess_train_val,
                                                  batch_size=32,
                                                  shuffle=True)

    classes, num_classes = get_enabled_classes(train_loader, preprocess_train_val)

    model = create_yolov5_pt(model_name="yolov5s", num_classes=num_classes, device=dev, pretrained=True)

    os.makedirs("models", exist_ok=True)
    writer = create_tb("tensorboard")

    # Initialize optimizer and scheduler
    optimizer = get_adam_optimizer_pt(model, learnrate=0.00001)
    #scheduler = get_reduce_lr_on_plateau_pt(optimizer, patience=20)

    for epoch in range(num_epochs):
        start_time = time.time()  # Record start time for the epoch
        loss_sum, loss_cnt = 0, 0

        batch: SampleContainer
        for batch in train_loader:
            optimizer.zero_grad()

            # load_data element
            input_data, annotations = batch.image_data.get().to(dev), batch.annotations.get(BoundingBox).to(dev)

            # predict element: forward pass
            predictions = predict_yolov5_pt(model, input_data)

            # calc_loss element: loss
            loss = calc_yolov5_loss_pt(p=predictions, targets=annotations, model=model, image_height=608, image_width=608)

            # optimize element: Backprop + Weights update
            back_prop_pt(loss, optimizer, model, clip_gradients=True)

            loss_sum += loss.item()
            loss_cnt += 1

        avg_loss = loss_sum / loss_cnt

        # visualize element
        show_loss_tb(avg_loss, epoch, writer=writer,name="train_loss")
        log_loss(avg_loss, epoch)

        # tune_params element
        #tune_learning_rate_pt(avg_loss, scheduler)

        # to obseve:
        current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
        print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")

        # Check if scheduler reduced LR this epoch
        #if hasattr(scheduler, 'num_bad_epochs'):
        #    print(f"Scheduler: {scheduler.num_bad_epochs} epochs without improvement")
        #    if scheduler.num_bad_epochs == 0:
        #        print("✓ Validation loss improved - patience reset")
        #    elif scheduler.num_bad_epochs >= scheduler.patience:
        #        print("🔻 LR REDUCED due to plateau!")

        # validate pipeline, if enough epochs have elapsed
        if (epoch % 2 == 0 and epoch != 0) or epoch == num_epochs - 1:
            model.eval()
            run_validation(valid_loader, model, epoch, working_dir, dev, writer=writer)
            model.train()

        #end_time = time.time()  # Record end time for the epoch
        #epoch_time = end_time - start_time  # Calculate duration
        #print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

    return

def _test(test_only: bool = False, wait: bool = False):
    """
    This method runs the training, validation, testing pipeline on the Fall Detection dataset.
    """
    torch.cuda.empty_cache()
    preprocess = [
        RemapLabels(mapping={'fall': 0, 'standing': 1, 'sitting': 2}, from_class_names=True),
        Resize(height=608, width=608, p=1),
    ]

    if get_cicd_test_type() == CICDTestType.FULL_COMPLETE:
        train_fn = run_training
    else:
        train_fn = partial(run_training, num_epochs=2)

    train_ds_info = FallDatasetInfo(
        images_dir="/media/public_data/temp/Phuong/fall_dataset/voc/train/img",
        annotations_dir="/media/public_data/temp/Phuong/fall_dataset/voc/train/xml",
        mean=get_pt_mean(),
        std=get_pt_std(),
        class_names=['fall', 'standing', 'sitting']
    )
    valid_ds_info = FallDatasetInfo(
        images_dir="/media/public_data/temp/Phuong/fall_dataset/voc/valid/img",
        annotations_dir="/media/public_data/temp/Phuong/fall_dataset/voc/valid/xml",
        mean=get_pt_mean(),
        std=get_pt_std(),
        class_names=['fall', 'standing', 'sitting']
    )
    test_ds_info = FallDatasetInfo(
        images_dir="/media/public_data/temp/Phuong/fall_dataset/voc/test/img",
        annotations_dir="/media/public_data/temp/Phuong/fall_dataset/voc/test/xml",
        mean=get_pt_mean(),
        std=get_pt_std(),
        class_names=['fall', 'standing', 'sitting']
    )

    if not test_only:
        train_fn(train_ds_info=train_ds_info, valid_ds_info=valid_ds_info, preprocess_train_val=preprocess)

    run_testing(test_ds_info=test_ds_info, preprocess_test=preprocess, wait=wait)
    if wait:
        wait_forever("Press stop or CTRL+C when finished examining the TensorBoard.")

if __name__ == "__main__":
    _test(test_only=False, wait=True)