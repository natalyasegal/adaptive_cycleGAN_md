
# Cycle GANs with Adaptive Texture Loss for Medical Image Domain Shift 
adaptive_cycleGAN_md

# Abstract
This work explored possibilities to shift embryo dataset distribution with the motivation of being able to combine a variety of small distribution-shifted datasets taken by various quality optics, which is the reality of the field. We adapted the domain of a low-quality embryo dataset (ED1) to match that of a high-quality dataset (ED4) created by Vitrolife Timelapse, utilizing Cycle GANs with texture preservation. The quality of the adaptation was assessed using a CNN classifier trained on ED4, yielding an AUC of 0.9026, balanced accuracy of 82.09%, and a Cohen Kappa of 0.6262 for the shifted ED1 dataset. Further evaluation on the ED3-shifted dataset showed superior results (AUC of 0.9676, balanced accuracy of 89.48%, the F1 score of 0.885, and the Kappa of 0.789) compared to ED3-grayscale, indicating promising avenues for future research.


Image example from the source domain:
![image](https://github.com/natalyasegal/adaptive_cycleGAN_md/assets/71938116/762bf556-3e63-4148-ab60-6fcacc48fb4f)


Why not to put too much strength on a texture loss:
![image](https://github.com/natalyasegal/adaptive_cycleGAN_md/assets/71938116/b9cfcdd2-cdbe-4109-a9c6-3eb127d507d4)




