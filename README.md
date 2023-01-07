# image_augmentation_pipeline
* 使用 imgaug 進行實作 (img_iaa.py), 更多 augmentation 方式可參考：
    > https://github.com/aleju/imgaug
* 加入 custom augmentation (img_custom.py), 每一次 image 均執行此 custom 隨機調整亮度與對比。 

## 1. indicators method
### 使用方式：
```python=
    img = cv2.imread('sample/sample.jpg')  
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= np.array(img, dtype=np.float32)
    img_origin=img.copy()
    
    Augmenation=train_augmentations()
    
    plt.figure(figsize=(20,20))
    for i in range(10):
        output=Augmenation(img)
        plt.subplot(1, 11, 1)
        plt.imshow(img_origin.astype(np.uint8))
        plt.axis('off')
        plt.subplot(1, 11, i+2)
        plt.imshow(output.astype(np.uint8))
        plt.axis('off')
    
    plt.show()
```
### output :
![result](result.png)

### 建立 generator 加入到 keras fit_generator 的方式可參考：
> https://github.com/aleju/imgaug/issues/66


