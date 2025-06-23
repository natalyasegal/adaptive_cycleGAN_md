
# Cycle GANs with Adaptive Texture Loss for Medical Image Domain Shift 
adaptive_cycleGAN_md

# Abstract
In this study, we address the challenge of distributional shifts in embryo imaging datasets, a common issue in clinical embryology due to variability in optical imaging quality across clinics and systems. We propose a domain adaptation approach to align low-quality embryo data (ED1) with a high-quality dataset (ED4) acquired using the Vitrolife time-lapse system. Using CycleGANs with texture-preserving constraints, we adapted ED1 and evaluated the quality of domain transfer on the blastocyst vs. non-blastocyst classification task. A convolutional neural network (CNN) trained on ED4 achieved an AUC of 0.9026, balanced accuracy of 82.09%, and Cohen’s Kappa of 0.6262 when tested on the adapted ED1. Applying the same method to ED3 led to further improvement (AUC = 0.9676, balanced accuracy = 89.48%, F1 score = 0.885, Kappa = 0.789), outperforming the original ED3 grayscale version. These results demonstrate the potential of domain adaptation to harmonize heterogeneous embryo datasets and enhance the robustness of machine learning models for embryo viability assessment.

# Introduction
Inspired by Kanakasabapathy et al. [1], our study addresses the scarcity of clean, annotated, and high-quality image data for medical imaging classification, where class rarity often leads to data imbalance. Challenges are compounded by the low number of annotated samples, particularly for new or low-end devices, and by equipment diversity that introduces image distribution shifts. For instance, embryo images used in assessment and grading vary significantly due to differences in the time-lapse systems and the optics of microscopes.
Kanakasabapathy et al. [1] proposed using MD-Nets and MD-Nets No Source (NoS) to address these issues. While MD-Nets offer a valuable approach for embryo imaging, their complex architecture complicates updates with newer methods. The MD-Nets NoS, untested by the authors on embryo datasets, initially showed poor performance (AUC 0.5) but improved to AUC 0.68 (on the basolocyst classification task) after modifying the clustering component to GMM.
Building on these insights, our research focuses on shifting the distribution of embryo datasets to align with data from more commonly used devices. This domain shifting enables the application of classifiers trained on data from established devices to newer, possibly more economical ones, without the need to collect and label large datasets from these new devices. Additionally, this approach facilitates the merging of multiple small datasets into a larger, unified dataset for more efficient classifier training.

![image](https://github.com/user-attachments/assets/6fb6c9fb-1a38-45ca-b792-d699a64b5fa1)

Fig. 1a Sample images from the datasets


# Data
![image](https://github.com/user-attachments/assets/b6998203-ebd9-4291-b6ae-95c847f32a44)

Fig. 1b, Datasets

Datasets ED1-4 (Fig. 1) can be downloaded from OSF: Medical Domain Adaptive Neural Networks. 
The dataset (ED4) contains 2440 images of embryos (taken by a Virtolife embryoscope Timelapse) from 374 patients captured at 113h after insemination (the average time of blastulation). The dataset contains a single image per embryo (no additional focal planes, no temporal component) and labels indicating whether the embryo in the image is a blastocyst (1,556 blastocysts and 884 non-blastocysts). The images come from high-quality optics. The dataset (ED1) comprises 296 images of embryos captured using a phone-based microscope with low-quality optics. 206 of these images are labeled as blastocysts, and the remaining 90 are labeled as non-blastocysts. The dataset (ED3) contains 258 images of embryos, 117 of which are labeled as blastocysts and the remaining 141 as non-blastocysts. In our experiments, we focus on datasets ED4, ED1, and ED3.

# Metrics
It is essential to note the imbalance between classes in the datasets above, so that when evaluating the distribution, we can assess the shift results with a classifier. We need metrics that account for the imbalance, such as Kappa and balanced accuracy, as well as metrics that perform relatively well with imbalanced datasets, like AUC. In this particular problem domain, we are equally interested in avoiding mistakes in both classes.
We are trying to balance the confusion matrix by ensuring a similar number of mistakes in every category.

# Cycle GAN architecture
In typical embryo datasets, as well as in most medical datasets, there are no pairs of the same embryo captured by different optics, so we decided to use Cycle GAN instead of Pix2Pix. 

We based our implementation on the classical cycle gan [9] and added texture-preserving loss. In medical images, and particularly in embryo images, texture plays a crucial role in detecting features and classifying conditions. We decided to use EfficientNet b7 texture loss after experimenting with a couple of other options (like VGG texture loss).

![image](https://github.com/user-attachments/assets/f53d874e-2924-4841-8ae2-3e63d6813637)

Fig. 2 Resulting ED1-distribution shifted image, with texture preservation (left) and without (middle - different embryo). In the middle, the internal texture pattern is completely lost compared to the image it originated from (right).

Schematic architecture of the cycle Gans - unpaired image-to-image translation. In our case, ED1 or ED3 is the “from” dataset, and ED4 is the “to” dataset. The adversarial loss also includes a texture-preserving loss.

Schematic architecture of the Cycle-Gans:

![image](https://github.com/user-attachments/assets/8ec957a1-8d6e-4042-adba-a8981fd1a008)

Fig. 3. Schematic architecture of the cycle Gans. In our case, ED1 is the “from” dataset, and ED4 is the “to” dataset. Based on a figure from the internet. The adversarial loss also includes a texture-preserving loss. (Original image is from the public domain)


# Results
We explored possibilities to shift the embryo dataset distribution with the goal of eventually combining a variety of small distribution-shifted datasets obtained using various quality optics, which is the current reality in the field. Initially, we enhanced the baseline classifier by replacing Resnet-50 and Xception [1] with EfficientNet-b7, as it demonstrated superior performance. This upgrade increased the average accuracy on the ED4 test set from 89.78% - the highest result reported in the paper among five CNN architectures—to 96.55% using EfficientNetB7 across five different splits with 5-fold cross-validation (Table 1).

Table 1: 
Table 1: Comparison of average performance between supervised CNNs (Xception /ResNet-50) [1] and EfficientNetB7 (using pre-trained weights) on the ED4 dataset, averaging test results from 5-fold


![image](https://github.com/user-attachments/assets/16747323-13f8-476d-877b-18f076f80bed)


Secondly, we evaluated the effectiveness of a Cycle GAN in domain adaptation by using a classifier trained on ED4 embryo images to distinguish between blastocysts and non-blastocysts. This classifier was tested on ED1 and ED3 datasets, which had been shifted to resemble the ED4 domain while preserving the texture of the original images using an adaptive Cycle GAN. The aim was to assess the GAN's performance in domain adaptation. EfficientNetB7-based CNN classifier shows an outstanding performance on a test set (handout that was not used for training) of ED4 with AUC of 0.99, balanced accuracy of 94.9, and F1 of 0.95 after only 10 epochs of the classifier training.
We transferred the domain of a low-quality embryo dataset (ED1) to match a high-quality dataset (ED4) using Cycle GANs with texture preservation loss, trained for 30 epochs. The ED1-shifted was evaluated using a CNN classifier trained on ED4, achieving an AUC of 0.903, a balanced accuracy of 82.09%, an F1 score of 0.88, and a Kappa of 0.626. Although these results are superior to those from similar classifiers used on unmodified ED1, converting ED1's color data to luma yielded even better metrics: an AUC of 0.957, a balanced accuracy of 87.96%, an F1 score of 0.93, and a Kappa of 0.766.

![image](https://github.com/user-attachments/assets/f062ea9e-da55-4064-ba24-470e1f1952ab)

Fig. 4: ROC-AUC Curve for the EfficientNet B7-Based Classifier on Domain-Shifted ED1 (test set). This classifier, trained on the target domain of the ED4 dataset, is used for assessing blastocyst identification.

After 10 epochs of the Cycle GAN, the ED3-shifted dataset performed better on the blastocyst classifier than the ED3-grayscale. For the ED3-shifted dataset, the AUC was 0.9676, balanced accuracy was 89.48%, the F1 score was 0.8851, and the Kappa was 0.789. In comparison, the ED3-grayscale achieved an AUC of 0.936, balanced accuracy of 85.64%, F1 score of 0.8439, and Kappa of 0.7113 (see Table 2).

Table 2: Metrics received for blastocyst vs. non-blastocyst classifier on original and shifted datasets.

![image](https://github.com/user-attachments/assets/673136a3-c9e6-42cf-bef0-1c72d8db975d)


We conducted our evaluation focusing on a single classification task. While various known classification tasks of differing complexity exist for embryo images, a more comprehensive approach would involve using a series of these tasks for thorough evaluation. We did not undertake these tasks here, as they were beyond the project's scope and would have required significant costs for labeling.
The results from the Cycle GAN-based approach are significant because they demonstrate the potential to consolidate small medical datasets into a larger, unified dataset for consistent labeling and classifier training.


--------------

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




