import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from utility import mask2rle


from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F
from utility import dice_coeff
import sys

np.set_printoptions(threshold=sys.maxsize)



def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    dir_base = "/UserData/"
    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    test_dice = []

    pred_rle_list = []
    target_rle_list = []
    ids_list = []
    dice_list = []
    i = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            images, targets, sentences, attentions, row_ids = data
            images, targets, sentences, attentions = images.to(device), targets.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            #print(f"setences size: {sentences.size()}")
            attentions = attentions.squeeze(1)
            target_gpu = targets
            target = targets.cpu().data.numpy()
            #for j in range(sentences.size(-1)):
            for j in range(1):
                #print(f"j: {j}")
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(images, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    #print(f"setences size: {sentences.size()}")
                    #print(f"attentions size: {attentions.size()}")
                    #output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])
                    outputs = model(images, sentences, l_mask=attentions)
                #print(f"target: {target}")
                #print(type(target_gpu))
                #print(f"output type: {type(output)}")
                # """

                for j in range(0, outputs.shape[0]):
                    #print(f"output size: {output[0].size()}")
                    #print(f"target size: {target[0].size()}")
                    dice = dice_coeff(outputs[0], target_gpu[0])
                    dice = dice.item()
                    #print(f"dice index : {len(test_dice)} with value: {dice}")
                    # if torch.max(output[i]) == 0 and torch.max(target[i]) == 0:
                    #    dice = 1
                    test_dice.append(dice)


                    #print(f"outputs size: {outputs.size()}")
                    #print(f"outputs size: {outputs[0].size()}")
                    #print(f"targets size: {targets.size()}")

                    output_item = outputs[j].cpu().data.numpy().argmax(0)
                    target_item = targets[j].cpu().data.numpy()
                    #print(f"output item size: {output_item.shape}")
                    #output_mask = output_item[0,:,:] + output_item[1,:,:]
                    #output_mask = output_item[1, :, :]
                    #output_mask = np.expand_dims(output_mask, 0)
                    #print(f"output_mask: {output_item.shape}")
                    #print(f"type: {type(output_item)}")
                    #print(f"full output: {output_item}")
                    pred_rle = mask2rle(output_item)
                    #print(f"target_item size: {target_item.shape}")

                    target_rle = mask2rle(target_item)
                    print(f"index: {i*8 + j}")
                    ids_example = row_ids[i * 8 + j]

                    pred_rle_list.append(pred_rle)
                    target_rle_list.append(target_rle)
                    ids_list.append(ids_example)
                    dice_list.append(dice)
                    """
                    # print(f"Target size: {targets.size()}")
                    target_np = targets.cpu().detach().numpy()
                    target_np = target_np[j, :, :]
                    max = np.amax(target_np)
                    target_np = (target_np * 255) / max
                    fullpath = os.path.join(dir_base,
                                            'Zach_Analysis/dgx_images/model_output_comparisons/lavt/targets/' + str(
                                                ids_example) + '.png')
                    #cv2.imwrite(fullpath, target_np)

                    # print(f"outputs: {outputs.size()}")
                    output = outputs.cpu().detach().numpy()
                    output = output[j, :, :]
                    max = np.amax(output)
                    output = (output * 255) / max
                    fullpath = os.path.join(dir_base,
                                            'Zach_Analysis/dgx_images/model_output_comparisons/lavt/outputs/' + str(
                                                ids_example) + '.png')
                    #cv2.imwrite(fullpath, output)

                    # print(f"images size: {images.size()}")

                    # image = images.cpu().detach().numpy()
                    image = images[j, 0, :, :]
                    image = image.cpu().detach().numpy()
                    # images = images[0, :, :]
                    fullpath = os.path.join(dir_base,
                                            'Zach_Analysis/dgx_images/model_output_comparisons/lavt/images/' + str(
                                                ids_example) + '.png')
                    #cv2.imwrite(fullpath, image)

                    #img_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    # print(np.sum(model_output) / 255)
                    #target_batch_unnorm = targets.cpu().detach().numpy()
                    #img_overlay[:, :, 1] += (
                    #            target_batch_unnorm[j, 0, :, :] * (255 / 3) / np.amax(target_batch_unnorm[j, 0, :, :]))
                    #fullpath = os.path.join(dir_base,
                    #                        'Zach_Analysis/dgx_images/model_output_comparisons/lavt/target_overlay/' + str(
                    #                            ids_example) + '.png')
                    #cv2.imwrite(fullpath, img_overlay)

                    #img_overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    model_output = outputs.cpu().detach().numpy()
                    output_overlay = (model_output[j, :, :] * 255 / 3) / np.amax(model_output[j, :, :])

                    # print(output_overlay.shape)
                    np.squeeze(output_overlay)
                    # print(output_overlay.shape)
                    #img_overlay[:, :, 1] += output_overlay[:, :]
                    # img_overlay[:, :, 1] += output_overlay[j, 0, :, :]

                    # print(f"model_output: {np.shape(model_output)}")
                    fullpath = os.path.join(dir_base,
                                            'Zach_Analysis/dgx_images/model_output_comparisons/lavt/output_overlay/' + str(
                                                ids_example) + '.png')
                    #cv2.imwrite(fullpath, img_overlay)
                    """
                i += 1

                # """

                outputs = outputs.cpu()
                output_mask = outputs.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del images, target, sentences, attentions, outputs, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding


    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)
    print(f"Final Test Dice: = {np.average(test_dice)}")

    test_df_data = pd.DataFrame(pd.Series(ids_list))
    # test_df_data["ids"] = pd.Series(ids_list)
    test_df_data["dice"] = pd.Series(dice_list)
    test_df_data["target"] = pd.Series(target_rle_list)
    test_df_data["prediction"] = pd.Series(pred_rle_list)

    filepath = os.path.join("UserData/Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/higher_res_for_paper/LAVT_v50/prediction_dataframe" + str(test) + '.xlsx')
    test_df_data.to_excel(filepath, index=False)

    return np.average(test_dice)



def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def test_main(args, dataset_test):
    device = torch.device(args.device)
    #dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    #print(args.model)
    #print(args.resume)
    #single_model = segmentation.__dict__[args.model](pretrained='./checkpoints/model_best_lavt_seed98.pth', args=args)
    single_model = segmentation.__dict__[args.model](pretrained=args.resume, args=args)

    print(f"path: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'])
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None
    print(f"using a bert model: {bert_model}")
    acc = evaluate(model, data_loader_test, bert_model, device=device)
    return acc


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    print("iniate main still called")
    main(args)
