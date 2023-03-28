# LOOK: A Wrapper For A Visual Attention Model

Simple wrapper code to run inference on images using the model and code from the [Looking Repo from VITA-EPFL](https://github.com/vita-epfl/looking).

The Looking Repo is an official implementation of the paper [Do pedestrians pay attention? Eye contact detection for autonomous driving](https://arxiv.org/abs/2112.04212)

![alt text](https://github.com/cwittwer/look/blob/main/images/people-walking-on-pedestrian-lane-during-daytime.pedictions.png)

Image taken from : https://jooinn.com/people-walking-on-pedestrian-lane-during-daytime.html . Results obtained with the model trained on JackRabbot, Nuscenes, JAAD and Kitti. The model file is available at ```models/predictor``` and can be reused for testing with the predictor. 

## Table of contents

- [Requirements](#requirements)
- [Predictor](#predictor)
  * [Example command](#example-command-)
- [Cite VITA-EPFL's work](#cite-VITA-EPFL's-work)


## Requirements

Use ```3.6.9 <= python < 3.9```. Run ```pip3 install -r requirements.txt``` to get the dependencies

## Custom Training And Model Evaluation

Please refer to the [original repo](https://github.com/vita-epfl/looking) for training custom models and evaluation of models.

## Predictor

<img src="https://github.com/cwittwer/look/blob/main/images/kitti.gif" data-canonical-src="https://github.com/cwittwer/look/blob/main/images/kitti.gif" width="1238" height="375" />

Get predictions from the pretrained model using any image with the predictor API. The API extracts the human keypoints on the fly using [OpenPifPaf](https://openpifpaf.github.io/intro.html). **The predictor supports eye contact detection using human keypoints only.**

## Run Inference
<ul>
  <li>Create an instance of the Predictor (with default or custom settings)</li>
      
      import look
      pred = look.Predictor()
      
  <li>Run predict on an RGB OpenCV format image</li>
        
      pred.predict(image)
      
  <li>Call for results, in either data or an image with overlayed information</li>
        
      output_image = pred.get_output_image()
      

</ul>

## Predictor Initialization Parameters

| Parameter                 |Default Value   |Description   |
| :------------------------ |:---------------|:-------------|
| ```transparency``` | ```0.4``` | transparency of the overlayed poses ```float``` |
| ```looking_threshold``` | ```0.5``` | eye contact threshold ```float``` |
| ```mode``` | ```joints``` | prediction mode ```string``` |
| ```device``` | ```'0'``` | CUDA device ```string``` |
| ```pifpaf_ver``` | ```shufflenetv2k30``` | PIFPAF ARG: backbone model to use ```string``` |
| ```model_path``` | ```models/predictor``` | To use custom trained model ```string``` |
| ```batch_size``` | ```1``` | PIFPAF ARG: processing batch size ```int``` |
| ```long_edge``` | ```None``` | PIFPAF ARG: rescale the long side of the image (aspect ratio maintained) ```int``` |
| ```loader_workers``` | ```None``` | PIFPAF ARG: number of workers for data loading ```int``` |
| ```disable_cuda``` | ```False``` | PIFPAF ARG: disable CUDA ```bool``` |


### Example Code To Run:

  ```
  import cv2
  import look

  pred = look.Predictor()

  cap = cv2.VideoCapture(0)

  if cap.isOpened() == False:
      print("Camera feed is not open")
      exit()

  width = int(cap.get(3))
  height = int(cap.get(4))

  print(f'Image Size: {width} X {height}')

  while True:
      ret, frame = cap.read()

      if ret == True:
          pred.predict(frame)
          frame = look.get_output_image()

          cv2.imshow('frame', frame)

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
      
      else:
          print('Could not get frame from video')
          break

  cap.release()
  cv2.destroyAllWindows()
  ```

## Credits

Credits to [OpenPifPaf](https://openpifpaf.github.io/intro.html) for the pose detection part.

## Cite VITA-EPFL's work

If you use our work for your research please cite VITA-EPFL :) 

```
@misc{belkada2021pedestrians,
      title={Do Pedestrians Pay Attention? Eye Contact Detection in the Wild}, 
      author={Younes Belkada and Lorenzo Bertoni and Romain Caristan and Taylor Mordan and Alexandre Alahi},
      year={2021},
      eprint={2112.04212},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
