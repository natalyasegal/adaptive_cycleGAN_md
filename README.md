
# Cycle GANs with Adaptive Texture Loss for Medical Image Domain Shift 
adaptive_cycleGAN_md

# Abstract
In this study, we address the challenge of distributional shifts in embryo imaging datasets, a common issue in clinical embryology due to variability in optical imaging quality across clinics and systems. We propose a domain adaptation approach to align low-quality embryo data (ED1) with a high-quality dataset (ED4) acquired using the Vitrolife time-lapse system. Using CycleGANs with texture-preserving constraints, we adapted ED1 and evaluated the quality of domain transfer on the blastocyst vs. non-blastocyst classification task. A convolutional neural network (CNN) trained on ED4 achieved an AUC of 0.9026, balanced accuracy of 82.09%, and Cohen’s Kappa of 0.6262 when tested on the adapted ED1. Applying the same method to ED3 led to further improvement (AUC = 0.9676, balanced accuracy = 89.48%, F1 score = 0.885, Kappa = 0.789), outperforming the original ED3 grayscale version. These results demonstrate the potential of domain adaptation to harmonize heterogeneous embryo datasets and enhance the robustness of machine learning models for embryo viability assessment.

# Data
![image](https://github.com/user-attachments/assets/b6998203-ebd9-4291-b6ae-95c847f32a44)

Fig. 5 Datasets

Some of the data used in the paper is available here: OSF: Medical Domain Adaptive Neural Networks. The dataset (ED4) contains 2440 images of embryos (taken by a Virtolife embryoscope Timelapse) from 374 patients captured at 113h after insemination (the average time of blastulation). Development stage annotations are not included with the dataset, which consists of a single image per embryo (no additional focal planes, no temporal component). The frames are annotated as blastocyst/not blastocyst.
Labels. Under all data, there are five directories in ED1 and ED4. A directory named “1” contained images that represent a no_blast category, and directories 2-5 represent the blast category. In ED3, there are two directories: “1” - no blast, “2” - blast.

Dataset 2: We have found an additional dataset relevant to embryo classification that was not used in the paper. It has development stage labels, so we can tell whether each image is a blastocyst. This dataset of 704 wells (embryos) can be downloaded from https://zenodo.org/records/6390798, and its description is: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9120221/ , one of the papers using this dataset: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9120221/# The daaset includes 704 distinct wells (embryos) with 3-5 days' worth of image (jpeg) information per well; images are taken in timelapse approximately once an hour (but sampling rate can vary). Development stages labels in embryo_dataset_annotations.tar.gz. There are 7 focal planes. We are most interested in the central focal plane and the ones close to it, preferably balancing the amount of data taken from every side. There are multiple images per stage (30-40 images) are present, allowing for the exploration of the temporal component with this data. In annotated Excel files, we are provided with a range for each development stage. For blastulation, please look for tB.

In our experiments, we focused on datasets ED4 (target domain) and ED1, ED3.

# Results of the Cycle Gan experiment
First, we enhance the baseline by refining the results of the supervised CNN-based 2-class classifier, utilizing EfficientNet-b7 instead of ResNet-50 and Xception, which yielded the best performance in the paper [1]. This improved the average accuracy on ED4 from 89.78% (the best result in the paper out of 5 CNN architectures) to 96.55% (Grayscale) and 95.38% (RGB) with EfficientNetB7. F1 score is 0.97 and 0.96, respectively. (Table1) 

Then, we test distribution-shifted datasets using this classifier. The distribution shift is implemented using Cycle GANs that are optimized to preserve the texture and aspect ratio of the original images. The motivation to preserve the texture can be seen in Fig. 7

![image](https://github.com/user-attachments/assets/d7fc562f-d36b-4733-856a-c38ea23078f7)

Fig. 7 Motivation for choosing Cycle Gans with texture preservation as part of the loss. Missing pairs of images to choose Pix to Pix [9]. If we do not put an effort to preserve the texture, the actual embryo image gets completely random.


We explored possibilities to shift the embryo dataset distribution, motivated by the goal of eventually combining a variety of small distribution-shifted datasets obtained using various quality optics, which is the current reality in the field. As the first step towards this goal, we shifted the domain of the very low-quality embryo dataset (ED1) to the domain of the highest-quality embryo dataset (ED4), using cycle GANs with texture preservation loss and evaluated the quality of the results by using CNN classifier (blastocyst or not) trained on ED4 to perform classification on ED1 with the following results:  an accuracy of 72%, balanced accuracy of 73.6%, and an AUC of 0.81. (Fig. 8-9) This part of the research is also promising and could be continued beyond the project.

![image](https://github.com/user-attachments/assets/174ad845-efb2-46da-ae98-222bb7953e9f)

Fig. 8 Results of evaluating the distribution shifted the ED1 dataset on the CNN classifier that was trained on the ED4 dataset. Evaluations on the original ED4 and ED1 are given for comparison.

![image](https://github.com/user-attachments/assets/8424fa3e-d52b-41c8-bb3f-e1dfbc00e9bd)

Fig. 9 AUC-ROC curve of domain-shifted ED1 performance on an efficientNetb7-based classifier that was trained in ED4

Table 1: Comparison of the results obtained by the best-performing supervised CNN (Xception and ResNet59)  to  EfficientnetB7 (Grayscale) proposed by us for the same task. Improve the classification baseline of the ED4 dataset with CNN. For that, we use EfficientNet-B7 grayscale (starting from pre-trained weights). On the ED4 handout test set of 238 samples with 5-fold cross-validation, using Grayscale images, we get the following results Mean Accuracy: 0.9655, Standard Deviation: 0.0108, Coefficient of Variation (CV): 1.11%, Accuracies: [0.9622, 0.9664, 0.9496, 0.9832, 0.96639], AUCs: [0.9915, 0.9981, 0.9832, 0.9991, 0.9923], Balanced Accuracies: [0.9710,0.9697,0.9459, 0.9824, 0.9632], F1 Scores: [0.9701,0.9688,0.9580, 0.9862, 0.9722] per fold. “-’ stands for the missing information in paper [1].

![image](https://github.com/user-attachments/assets/d89ae214-186f-4488-afaa-d8c97467f2bd)

Schematic architecture of the Cycle-Gans:

![image](https://github.com/user-attachments/assets/8ec957a1-8d6e-4042-adba-a8981fd1a008)

Fig. 10. Schematic architecture of the cycle Gans. In our case, ED1 is the “from” dataset, and ED4 is the “to” dataset. Based on a figure from the internet. The adversarial loss also includes a texture-preserving loss.


# Relevant repositories:
Cycle GAN reference code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
The codes and algorithms developed for this study [1], in particular, MD-nets
and its variants are available on GitHub (https://github.com/shafieelab/Medical-Domain-Adaptive-Neural-Networks).

# References:
[1] Kanakasabapathy, M.K., Thirumalaraju, P., Kandula, H., et al. (2021). Adaptive adversarial neural networks for the analysis of lossy and domain-shifted datasets of medical images. Nat Biomed Eng, 5, 571–585. https://doi.org/10.1038/s41551-021-00733-w. Link

[2] DANNs: Domain - Adversarial Adaptation. Link

[3] CDANs: Conditional Adversarial Domain Adaptation. Link

[4] Xception: Deep Learning With Depthwise Separable Convolutions. Link

[5] Md Nets: Learning Multi-Domain Convolutional Neural Networks for Visual Tracking. Hyeonseob Nam, Bohyung Han; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4293-4302.

[6] Kragh, M.F., Rimestad, J., Berntsen, J., Karstoft, H. (2019). Automatic grading of human blastocysts from time-lapse imaging. Computers in Biology and Medicine, 115, 103494. ISSN 0010-4825. https://doi.org/10.1016/j.compbiomed.2019.103494. Link

[7] Tan, M., & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Google Research.

[8] Zhu, J.-Y., Park, T., Isola, P., & Efros, A.A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. Proceedings of the IEEE International Conference on Computer Vision (ICCV). 

[9] Zhu, J.-Y., Park, T., Isola, P., & Efros, A.A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. Proceedings of the IEEE International Conference on Computer Vision (ICCV).








# Appendix A

Image example from the source domain:
![image](https://github.com/natalyasegal/adaptive_cycleGAN_md/assets/71938116/762bf556-3e63-4148-ab60-6fcacc48fb4f)


Why not to put too much strength on a texture loss:
![image](https://github.com/natalyasegal/adaptive_cycleGAN_md/assets/71938116/b9cfcdd2-cdbe-4109-a9c6-3eb127d507d4)




