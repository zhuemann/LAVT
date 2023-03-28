import os
import pandas as pd
import torchvision.transforms as T
from sklearn import model_selection
from transformers import AutoTokenizer
from candid_dataloader import TextImageDataset

def candid_data_setup(seed):
    #seed = 117
    dir_base = "/UserData/"
    #dataframe_location = os.path.join(dir_base,
    #                                  'Zach_Analysis/candid_data/pneumothorax_with_multisegmentation_positive_text_df.xlsx')

    #df = pd.read_excel(dataframe_location, engine='openpyxl')
    #print(df)
    #df.set_index("image_id", inplace=True)

    #train_df, test_valid_df = model_selection.train_test_split(
    #    df, train_size=.8, random_state=seed, shuffle=True  # stratify=df.label.values
    #)
    # Splits the test and valid sets in half so they are both 10% of total data
    #test_df, valid_df = model_selection.train_test_split(
    #    test_valid_df, test_size=.2, random_state=seed, shuffle=True
        # stratify=test_valid_df.label.values
    #)
    #save_location = "/UserData/Zach_Analysis/git_multimodal/lavt/LAVT/checkpoints"
    #test_dataframe_location = os.path.join('./checkpoints/pneumothorax_testset_df_seed' + str(117) + '.xlsx')
    #print(test_dataframe_location)
    #test_df.to_excel(test_dataframe_location, index=True)
    #test_dataframe_location = os.path.join(save_location,
    #                                       'pneumothorax_testset_df_seed' + str(config["seed"]) + '.xlsx')


    train_location = os.path.join("/UserData/Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/higher_res_for_paper/dataframe_saving/seed"+str(seed)+"/pneumothorax_train_df_seed"+str(seed)+".xlsx")
    train_df = pd.read_excel(train_location, engine='openpyxl')
    train_df.set_index("image_id", inplace=True)

    valid_location = os.path.join("/UserData/Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/higher_res_for_paper/dataframe_saving/seed"+str(seed)+"/pneumothorax_valid_df_seed"+str(seed)+".xlsx")
    valid_df = pd.read_excel(valid_location, engine='openpyxl')
    valid_df.set_index("image_id", inplace=True)

    test_location = os.path.join("/UserData/Zach_Analysis/result_logs/candid_result/image_text_segmentation_for_paper/higher_res_for_paper/dataframe_saving/seed"+str(seed)+"/pneumothorax_testset_df_seed"+str(seed)+".xlsx")
    test_df = pd.read_excel(test_location, engine='openpyxl')
    test_df.set_index("image_id", inplace=True)

    #print(test_dataframe_location)
    #test_df.to_excel(test_dataframe_location, index=True)
    # print(df)
    IMG_SIZE = 480
    #transforms_resize = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.PILToTensor()])
    #output_resize = transforms.Compose([transforms.Resize((1024, 1024))])
    #transforms_resize = T.Compose([T.Resize((1024, 1024))])
    transforms_resize = T.Compose([T.Resize((480, 480))])
    #def get_transform(args):
    transforms_candid = T.Compose([T.Resize((480, 480)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        #return T.Compose(transforms)

    #print("train_df")
    #print(train_df)
    #print("valid df")
    #print(valid_df)

    bert_path = os.path.join(dir_base, 'Zach_Analysis/models/bert/')
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    training_set = TextImageDataset(train_df, tokenizer, 512, mode="train", transforms=transforms_candid,
                                    resize=transforms_resize, dir_base=dir_base, img_size=IMG_SIZE, wordDict=None,
                                    norm=None)
    valid_set = TextImageDataset(valid_df, tokenizer, 512, mode="train", transforms=transforms_candid, resize=transforms_resize,
                                 dir_base=dir_base, img_size=IMG_SIZE, wordDict=None, norm=None)
    test_set = TextImageDataset(test_df, tokenizer, 512, mode="train", transforms=transforms_candid, resize=transforms_resize,
                                dir_base=dir_base, img_size=IMG_SIZE, wordDict=None, norm=None)



    return training_set, valid_set, test_set