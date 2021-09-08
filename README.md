# KNN-Averaging for Noisy Multi-Objective Optimisation
(Supporting Submission for the QUATIC 2021 Paper for kNN-Averaging)

**Authors**  
[Stefan Klikovits](https://klikovits.net) and [Paolo Arcaini](http://group-mmm.org/~arcaini/);  
National Institute of Informatics, Tokyo, Japan   
Email: `{lastname}@nii.ac.jp`  

**Venue**  
[14th International Conference on the Quality of Information and Communications Technology (QUATIC)](https://2021.quatic.org/)  
*Quality in Cyber-physical Systems* Track  
September 8-11, 2021    
~~Faro, Portugal~~ *Online*  

**Thanks**  
The authors are supported by ERATO HASUO Metamathematics for Systems Design Project (No. JPMJER1603), JST. Funding Reference number: 10.13039/501100009024 ERATO.
S. Klikovits is also supported by Grant-in-Aid for Research Activity Start-up 20K23334, JSPS.

---

(Preliminary) Bibtex:
```
@InProceedings{KlikovitsA2021knnAveraging
  author={Klikovits, Stefan and Arcaini, Paolo},
  title= {{KNN-Averaging for Noisy Multi-Objective Optimisation}},
  editor={Paiva, Ana C. R. and Cavalli, Ana Rosa and Ventura Martins, Paula and P{\'e}rez-Castillo, Ricardo},
  booktitle= {Proc. 14th Intl. Conf. on the Quality of Information and Communications Technology (QUATIC)},
  publisher= {Springer International Publishing},
  pages={503--518},
  series= {Communications in Computer and Information Science (CCIS)},
  volume={1439},
  isbn={978-3-030-85347-1},
  location= {Faro, Portugal (Online)},
  doi={10.1007/978-3-030-85347-1_36},
  year={2021}
}
```

## In this repository

* [`KlikovitsArcaini-KNNAvgForNoisyNoisyMOO.pdf`](./KlikovitsArcaini-KNNAvgForNoisyNoisyMOO.pdf) Preprint of the paper.
* [`requirements.txt`](./requirements.txt): Python dependencies. Install via `pip install -r requirements.txt`
* [`main.py`](./main.py): Entrypoint for search. To alter search settings, modify `create_settings_and_run()` function.
* [`knn_wrapper.py`](./knn_wrapper.py) Wrapper for [Pymoo](https://pymoo.org/) benchmark problems. Use via `knn_wrapper.wrap_problem(knn_wrapper.KNNAvgMixin, zdt.ZDT1, ...)`.
* [`output/`](./output) This will be where the search will put the output data (currently empty).
* [`result_plots/`](./results_plots/) Supporting data and result plots for the publication. Subfolders/filenames provide information about search settings
    * [`results_plots/KNNAvgMixin_ZDT1_V2/`](./results_plots/KNNAvgMixin_ZDT1_V2)  ZDT1 benchmark with 2 search variables
    * [`results_plots/KNNAvgMixin_ZDT1_V4/`](./results_plots/KNNAvgMixin_ZDT1_V4)  ZDT1 benchmark with 4 search variables
    * [`results_plots/KNNAvgMixin_ZDT1_V10/`](./results_plots/KNNAvgMixin_ZDT1_V10)  ZDT1 benchmark with 10 search variables
    * [`results_plots/KNNAvgMixin_ZDT2_V2/`](./results_plots/KNNAvgMixin_ZDT2_V2)  ZDT2 benchmark with 2 search variables
    * [`results_plots/KNNAvgMixin_ZDT2_V4/`](./results_plots/KNNAvgMixin_ZDT2_V4)  ZDT2 benchmark with 4 search variables
    * [`results_plots/KNNAvgMixin_ZDT2_V10/`](./results_plots/KNNAvgMixin_ZDT2_V10)  ZDT2 benchmark with 10 search variables
    * [`results_plots/KNNAvgMixin_ZDT3_V2/`](./results_plots/KNNAvgMixin_ZDT3_V2)  ZDT3 benchmark with 2 search variables
    * [`results_plots/KNNAvgMixin_ZDT3_V4/`](./results_plots/KNNAvgMixin_ZDT3_V4)  ZDT3 benchmark with 4 search variables
    * [`results_plots/KNNAvgMixin_ZDT3_V10/`](./results_plots/KNNAvgMixin_ZDT3_V10)  ZDT3 benchmark with 10 search variables
