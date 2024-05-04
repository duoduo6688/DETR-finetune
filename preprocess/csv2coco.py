import pandas as pd
import json
import os
import shutil

PARAMS = {
    "annotation_train_csv_path": "../../SKU110K_fixed/annotations/annotations_train.csv",
    "annotation_test_csv_path": "../../SKU110K_fixed/annotations/annotations_test.csv",
    "annotation_val_csv_path": "../../SKU110K_fixed/annotations/annotations_val.csv",
    "save_train_json_path": '../../SKU110K_fixed/data/custom_train.json',
    "save_test_json_path": '../../SKU110K_fixed/data/custom_test.json',
    "save_val_json_path": '../../SKU110K_fixed/data/custom_val.json'
}

def image(row):
    image = {}
    image["height"] = row.height
    image["width"] = row.width
    image["id"] = row.fileid
    image["file_name"] = row.filename
    return image

def category(row):
    category = {}
    category["supercategory"] = 'None'
    category["id"] = row.categoryid
    category["name"] = row[2]
    return category

def annotation(row):
    annotation = {}
    area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
    annotation["segmentation"] = []
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = row.fileid

    annotation["bbox"] = [row.xmin, row.ymin, row.xmax -row.xmin,row.ymax-row.ymin ]

    annotation["category_id"] = row.categoryid
    annotation["id"] = row.annid
    return annotation

def convert(annotation_csv_path, save_json_path, **kwargs):
    print(f"preprocess images csv: {annotation_csv_path}")

    # no column names in the top of csv file, we need add column names
    # filename,xmin,ymin,xmax,ymax,classname,width,height
    tmp_csv_path = annotation_csv_path+".tmp"
    shutil.copyfile(annotation_csv_path, tmp_csv_path)
    columns = "filename,xmin,ymin,xmax,ymax,classname,width,height"
    with open(tmp_csv_path, "r+") as f:
        old = f.read()
        f.seek(0)
        f.write(columns)
        f.write("\n")
        f.write(old)

    data = pd.read_csv(tmp_csv_path)
    print(f"csv size: {data.size}")

    # print(f"{data}")

    images = []
    categories = []
    annotations = []

    # category = {}
    # category["supercategory"] = 'none'
    # category["id"] = 0
    # category["name"] = 'None'
    # categories.append(category)

    data['fileid'] = data['filename'].astype('category').cat.codes
    data['categoryid']= pd.Categorical(data['classname'],ordered= True).codes
    data['categoryid'] = data['categoryid']+1
    data['annid'] = data.index

    for row in data.itertuples():
        annotations.append(annotation(row))

    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))

    catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
    for row in catdf.itertuples():
        # print(f"row: {row}")
        category = {}
        category["supercategory"] = 'none'
        category["id"] = row.categoryid
        category["name"] = row.classname
        categories.append(category)
        # categories.append(category(id=row.categoryid))

    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations

    json.dump(data_coco, open(save_json_path, "w"), indent=4)

    print(f"save to {save_json_path}\n")

    # clean tmp file
    os.remove(tmp_csv_path)

if __name__ == "__main__":
    convert(PARAMS["annotation_train_csv_path"], PARAMS["save_train_json_path"])
    convert(PARAMS["annotation_test_csv_path"], PARAMS["save_test_json_path"])
    convert(PARAMS["annotation_val_csv_path"], PARAMS["save_val_json_path"])
