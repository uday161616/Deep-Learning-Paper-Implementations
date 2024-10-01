# Implementation of the paper Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks


### Images generated from the CycleGAN model.

* Horse to Zebra

![Horse to Zebra](saved_images/zebra_600.png)
![Horse to Zebra](saved_images/zebra_1200.png)



* Zebra to Horse

![Zebra to Horse](saved_images/horse_200.png)
![Zebra to Horse](saved_images/horse_1200.png)

The shape is of the horse, but strips are still visible. Have to improve this!


* Trained for 150 epochs, with a learning rate of 1e-5 (eventhough the paper mentioned 2e-4). Should experiment with this. 
* Didn't use Identity loss.