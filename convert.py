import json

def json_convert(a):
    for item in a:
        item['filename'] = item['filename'][:-4]
    return a
    
    
def main():    
    gallery_path = "/media/lhx/lhx/depthmap_hdf5/json/gallery.json"
    probe_neutral_path = "/media/lhx/lhx/depthmap_hdf5/json/probe_neutral.json"
    frontial_path = "/media/lhx/lhx/depthmap_hdf5/json/frontial.json"

    gallery_save_path = "/media/lhx/lhx/depthmap_hdf5/json/Bos/gallery.json"
    probe_neutral_save_path = "/media/lhx/lhx/depthmap_hdf5/json/Bos/probe_neutral.json"
    frontial_save_path = "/media/lhx/lhx/depthmap_hdf5/json/Bos/frontial.json"  

    gallery = json_convert(json.load(open(gallery_path)))
    with open(gallery_save_path,"w") as f:
        json.dump(gallery,f)

    probe_neutral = json_convert(json.load(open(probe_neutral_path)))
    with open(probe_neutral_save_path,"w") as f:
        json.dump(probe_neutral,f)

    frontial = json_convert(json.load(open(frontial_path)))
    with open(frontial_save_path,"w") as f:
        json.dump(frontial,f)

if __name__ == "__main__":
    main()

