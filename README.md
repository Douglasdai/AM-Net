# AM-Net

Accurate segmentation for long nodules plays a key role in building the Computer-Aided Diagnosis (CAD) system. However, many existing long nodule segmentation frameworks only extract features from the original computed tomography (CT) scans, which leads their methods cannot have a high segmentation accuracy. In this work, we proposed an attribute-guided multi-window network (named AM-Net) for Lung Nodule Segmentation. We use nine medical features of lung nodules to guide the feature extraction of our AM-Net, which can significantly boost the segmentation performance. More importantly, we study the different representations of medical features in different window settings, and a multi-window (including default window, lung window, and mediastinal window) input is constructed for better feature learning. Each window setting corresponds to the extraction of the corresponding medical features. We evaluate the proposed AM-Net on the public dataset LIDC-IDRI. Extensive results suggest that our AM-Net achieves significant segmentation performance on lung nodule segmentation task. 

