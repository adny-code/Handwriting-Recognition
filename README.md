# Handwriting-Recognition

Recognition 0 ~ 9 handwriting image.

# file describe
```
root
├── hwrImage                # image process.
│   ├── __init__.py
│   └── image_handle.py
├── hwrModel                # model.
│   ├── __init__.py
│   └── knn_model.py
├── rsc                     # resource files.
│   ├── origin
│   │   ├── 0_xxx.bmp
│   │   ···
│   │   ├── 1_xxx.bmp
│   │   ···
│   │   ├── 2_xxx.bmp
│   │   ···
│   │   ···
│   |   └── 9_xxx.bmp
│   |   
│   └── test_pattern
│       ├── x_xxx.bmp
│       ···
│       └── x_xxx.bmp
└── main.py
```


# RUN
python -u main.py


# TODO
- Add other models.
- Implement NEW model.
- Recognition words.
